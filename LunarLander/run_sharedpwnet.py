import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

from collections import Counter
from copy import deepcopy
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import gumbel_softmax, cosine_similarity
from argparse import ArgumentParser
import os
from os.path import join
from itertools import combinations
from torch.distributions import Beta

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep
from model import ActorCritic
import datetime

from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("n_proto", type=int, default = 6, help="Number of prototypes to be learned")
parser.add_argument("n_slots", type=int, default = 2, help="Number of slots per class")
parser.add_argument("new_proto_init", nargs='?', default=False, const=True, type=bool, help='Specify new proto initialization argument')

args = parser.parse_args()

NUM_PROTOTYPES = args.n_proto
NUM_SLOTS_PER_CLASS = args.n_slots

SANITY_CHECK = False

NUM_ITERATIONS = 15
NUM_EPOCHS = 100
NUM_CLASSES = 4

LATENT_SIZE = 128
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
DEVICE = 'cuda'
delay_ms = 0
NUM_SIMULATIONS = 30

clst_weight = 0.008 # before: 0.08
sep_weight = -0.0008 # before: 0.008
l1_weight = 1e-5 #1e-4

name_file = "run_sharedpwnet"

current_date = datetime.date.today()
date = current_date.strftime("%d_%m_%Y")

# novel initialization
X_train = np.load('data/X_train.npy')
a_train = np.load('data/a_train.npy')
# actions space= [0,1,2,3]

action_states = {}
for action_id in range(NUM_CLASSES):
    action_states[action_id] = []
    
    for state, action in zip(X_train, a_train):
        if action == action_id:
            action_states[action_id].append(state)

prototypes = []

for action_id in range(NUM_CLASSES):
	prototypes.append(KMeans(NUM_SLOTS_PER_CLASS, n_init="auto").fit(action_states[action_id]).cluster_centers_)   # prototypes = [[p11,p12,p13],[p21,p22,p23],[p31,p32,p33],]

ordered_prototypes = []

for ps in zip(*prototypes):
	for p in ps:
		ordered_prototypes.append(p)   # ordered_prototypes = [p11,p21,p31,p12,p22,p32,p13,p23,p33]

init_prototypes = random.sample(ordered_prototypes, NUM_PROTOTYPES)

init_prototypes = torch.tensor(init_prototypes, dtype=torch.float32)

class SharedPwNet(nn.Module):
    def __init__(self):
        super(SharedPwNet, self).__init__()
        self.projection_network = nn.Sequential(
            nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
            nn.InstanceNorm1d(PROTOTYPE_SIZE),
            nn.ReLU(),
            nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
        )
        if args.new_proto_init:
            self.prototypes = nn.Parameter(init_prototypes, requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn((NUM_PROTOTYPES, LATENT_SIZE), dtype=torch.float32), requires_grad=True) # in pw-net: randn
        self.proto_presence = torch.zeros(NUM_CLASSES, NUM_PROTOTYPES, NUM_SLOTS_PER_CLASS)
        self.proto_presence = nn.Parameter(self.proto_presence, requires_grad=True)
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)
        
        self.prototype_class_identity = torch.zeros(NUM_SLOTS_PER_CLASS * NUM_CLASSES, NUM_CLASSES)  
        for i in range(NUM_SLOTS_PER_CLASS * NUM_CLASSES):
            self.prototype_class_identity[i, i // NUM_SLOTS_PER_CLASS] = 1
        # class_identity_layer = linear    
        self.class_identity_layer = nn.Linear(NUM_SLOTS_PER_CLASS * NUM_CLASSES, NUM_CLASSES, bias=False) 
        positive_one_weights_locations = torch.t(self.prototype_class_identity) # transpose
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = 0 # -0.5
        # to weight in the proper way the last linear layer
        self.class_identity_layer.weight.data.copy_(correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations)
        
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-5
    
    def prototype_layer(self, x):
        b_size = x.shape[0]
        transf_proto = list()
        for i in range(NUM_PROTOTYPES):
            #print(self.prototypes[i].view(1,-1).shape)
            transf_proto.append(self.projection_network(self.prototypes[i].view(1, -1)))
        latent_protos = torch.cat(transf_proto, dim=0) 
        
        p = latent_protos.T.view(1, PROTOTYPE_SIZE, NUM_PROTOTYPES).tile(b_size, 1, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE, 1).tile(1, 1, NUM_PROTOTYPES).to(DEVICE)    
                
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        # similarity function from Chen et al. 2019: to score the distance between state c and prototype p
        similarity = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return similarity # (batch, NUM_PROTOTYPES)
    
    def output_activations(self, out):
        return self.softmax(out)
    
    def forward(self, x, gumbel_scalar, tau):
        '''
        x (raw input) size: (batch, 256)
        '''
        if gumbel_scalar == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scalar, dim=1, tau = tau)
        
        x = self.projection_network(x)
        similarity = self.prototype_layer(x)
        
        mixed_similarity = torch.einsum('bp, cpn->bcn', similarity, proto_presence) # (batch, NUM_CLASSES, NUM_SLOTS_PER_CLASS)

        out1 = self.class_identity_layer(mixed_similarity.flatten(start_dim=1))
        
        out2 = self.output_activations(out1)
        return out2, x, similarity, proto_presence
    
    
def evaluate_loader(model, gumbel_scalar, loader, cce_loss, tau):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # size of imgs: [batch, 256], size of labels: [batch, 3]
            logits, _, _, _ = model(imgs, gumbel_scalar, tau)
            loss = cce_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_correct += sum(preds == labels).item()
            total += len(preds)
            total_loss += loss.item()
    model.train()
    return  (total_correct / total) * 100


start_val = 1.3
end_val = 10 **3 
epoch_interval = 30 # before: 10
alpha2 = (end_val / start_val) ** 2 / epoch_interval

def lambda1(epoch): return start_val * np.sqrt((alpha2 * (epoch))) if epoch < epoch_interval else end_val

def dist_loss(model, similarity, proto_presence, top_k, sep=False):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (LATENT_SIZE * 1 * 1)
    
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(basic_proto), index=idx)  # [b, p]
    inverted_distances, _ = torch.max((max_dist - similarity) * binarized_top_k, dim=1)  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost


if not os.path.exists('results/'):
    os.makedirs('results/')

if args.new_proto_init:
    results_file = f'results/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}_results_newinit.txt'
else:
    results_file = f'results/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}_results.txt'

print(f"NUM_PROTOTYPES: {NUM_PROTOTYPES}")
print(f"NUM_SLOTS: {NUM_SLOTS_PER_CLASS}")

with open(results_file, 'a') as f:
    f.write("--------------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}\n")
    f.write(f"NUM_PROTOTYPES: {NUM_PROTOTYPES}\n")
    f.write(f"NUM_SLOTS: {NUM_SLOTS_PER_CLASS}\n")
        
data_rewards = list()
data_accuracy = list()

for iter in range(NUM_ITERATIONS):
    
    MODEL_DIR = f'weights/{date}_{name_file}_model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    MODEL_DIR_ITER = f'weights/{date}_{name_file}_model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/iter_{iter}.pth'
    
    if args.new_proto_init:
        prototype_path = f'prototypes/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}_newinit/iter_{iter}/'
    else:
        prototype_path = f'prototypes/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/iter_{iter}/'
    if not os.path.exists(prototype_path):
        os.makedirs(prototype_path)
    
    with open(results_file, 'a') as f:
        f.write(f"ITERATION {iter}: \n")
        
    writer = SummaryWriter(f"runs/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/Iteration_{iter}")

    name='LunarLander_TWO.pth'
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    X_train = np.load('data/X_train.npy')
    a_train = np.load('data/a_train.npy')
    obs_train = np.load('data/obs_train.npy')
    
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.long)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

        
    #### Train
    model = SharedPwNet().eval()
    model.to(DEVICE)
    cce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    best_acc = 0.
    model.train()
    
    '''
    prototypes 
    proto_presence 
    projection_network.0.weight 
    projection_network.0.bias 
    projection_network.3.weight 
    projection_network.3.bias 
    class_identity_layer.weight 

    '''
    running_loss = running_loss_mse = running_loss_clst = running_loss_sep = running_loss_l1 =  running_loss_ortho = 0.
    for epoch in range(NUM_EPOCHS):

        model.eval()
        gumbel_scalar = lambda1(epoch)
        
        if epoch == 0:
            tau = 1
        elif (epoch + 1) % 8 == 0 and tau > 0.3:
            tau = 0.8 * tau   
        
        current_acc = evaluate_loader(model, gumbel_scalar, train_loader, cce_loss, tau)
        model.train()

        if current_acc > best_acc and epoch > NUM_EPOCHS-20:
            torch.save(model.state_dict(), MODEL_DIR_ITER) # saves model parameters
            best_acc = current_acc
        
        # prototype projection every 2 epochs
        if epoch >= 10 and epoch % 2 == 0 and epoch < NUM_EPOCHS-20:
            #print("Projecting prototypes...")
            transformed_x = list()
            model.eval()
            with torch.no_grad():
                for i in range(len(X_train)):
                    img = X_train[i]
                    img_tensor = torch.tensor(img, dtype=torch.float32).view(1, -1) # (1, 256)
                    _, x, _, _ = model(img_tensor.to(DEVICE), gumbel_scalar, tau)
                    # x è lo stato s dopo la projection network
                    transformed_x.append(x[0].tolist())
            transformed_x = np.array(transformed_x)
            
            list_projected_prototype = list()
            for i in range(NUM_PROTOTYPES):
                trained_p = model.projection_network(model.prototypes)
                trained_prototype_clone = trained_p.clone().detach()[i].view(1,-1)
                trained_prototype = trained_prototype_clone.cpu()
                knn = KNeighborsRegressor(algorithm='brute')
                knn.fit(transformed_x, list(range(len(transformed_x)))) # lista da 0 a len(transformed_x) - n of training data
                dist, transf_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
                projected_prototype = X_train[transf_idx.item()]# transformed_x[transf_idx.item()]
                list_projected_prototype.append(projected_prototype.tolist())
                
                if epoch == NUM_EPOCHS-20-2: 
                    print("I'm saving prototypes' images in prototypes/ directory...")
                    prototype_image = obs_train[transf_idx.item()]
                    prototype_image = Image.fromarray(prototype_image, 'RGB')
                    p_path = prototype_path+f'p{i+1}.png'
                    prototype_image.save(p_path)
                                            
            trained_prototypes = model.prototypes.clone().detach()
            tensor_projected_prototype = torch.tensor(list_projected_prototype, dtype=torch.float32) # (num_prot, 50)
            #model.prototypes = torch.nn.Parameter(tensor_projected_prototype.to(DEVICE))
            with torch.no_grad():
                model.prototypes.copy_(tensor_projected_prototype.to(DEVICE))
            model.train()
            
        # freezed prototypes and projection network, training only proto_presence (prototype assignment) + class_identity_layer (last layer)
        if epoch >= NUM_EPOCHS-20:
            for name, param in model.named_parameters():
                if "prototypes" in name: 
                    param.requires_grad = False 
                elif "projection_network" in name:
                    param.requires_grad = False 
                        
        for instances, labels in train_loader:
            optimizer.zero_grad()
                    
            instances, labels = instances.to(DEVICE), labels.to(DEVICE)
            logits, _, similarity, proto_presence = model(instances, gumbel_scalar, tau)
            
            loss1 = cce_loss(logits, labels) 
            # orthogonal loss --> for slots orthogonality: in this way successive slots of a class are assigned to different prototypes
            orthogonal_loss = torch.Tensor([0]).to(DEVICE)

            for c in range(model.proto_presence.shape[0]): # NUM_CLASSES
                list_p = list(range(1, model.proto_presence.shape[1]+1))
                for (i,j) in list(combinations(list_p, 2)):
                    s1 = model.proto_presence[c][i-1].view(1,-1)
                    s2 = model.proto_presence[c][j-1].view(1,-1)
                    sim = cosine_similarity(s1, s2, dim=1).sum()
                    orthogonal_loss += sim
            orthogonal_loss = orthogonal_loss / (NUM_SLOTS_PER_CLASS * NUM_CLASSES) - 1
            
            #print("labels: ", labels) # [batch size, int] tensor([2, 4, 5, 4, 0, 5, 4, 4, 3, 3, 3, 0, 0, 2, 5, 5, 5, 1, 1, 1, 0, 1, 4, 0,
            #0, 3, 4, 4, 4, 4, 3, 4, 4, 0, 2, 1, 0, 3, 3, 0], device='cuda:0')
            labels_p = labels.cpu().numpy().tolist()
            #label = [0/1/2/3/4/5]
            
            proto_presence = proto_presence[labels_p] # (?) labels_pp deve essere un vettore (batch size, classe) classe = 0,1,2
            inverted_proto_presence = 1 - proto_presence
            labels.to(DEVICE)
            
            clst_loss_val = dist_loss(model, similarity, proto_presence, NUM_SLOTS_PER_CLASS)  
            sep_loss_val = dist_loss(model, similarity, inverted_proto_presence, NUM_PROTOTYPES - NUM_SLOTS_PER_CLASS) 
            
            prototypes_of_correct_class = proto_presence.sum(dim=-1).detach()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            avg_separation_cost = torch.sum(similarity * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
            l1 = (model.class_identity_layer.weight * l1_mask).norm(p=1)
            loss = loss1 + clst_loss_val * clst_weight + sep_loss_val * sep_weight + l1 * l1_weight + orthogonal_loss 

            
            running_loss_mse += loss1.item()
            running_loss_clst += clst_loss_val.item() * clst_weight
            running_loss_sep += sep_loss_val.item() * sep_weight
            running_loss_l1 += l1.item() * l1_weight
            running_loss_ortho += orthogonal_loss.item() 
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
    
        print("Epoch:", epoch, "Running Loss:", running_loss / len(train_loader), "Current Accuracy:", current_acc)
        with open(results_file, 'a') as f:
            f.write(f"Epoch: {epoch}, Running Loss: {running_loss / len(train_loader)}, Current Accuracy: {current_acc}\n")

        writer.add_scalar("Running_loss: ", running_loss/len(train_loader), epoch)
        writer.add_scalar("Current_accuracy: ", current_acc, epoch)
        running_loss = running_loss_mse = running_loss_clst = running_loss_sep = running_loss_l1 =  running_loss_ortho = 0.
            
        scheduler.step()
    
    #states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []

    # Wrapper model with learned weights
    model = SharedPwNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR_ITER))
    model.to(DEVICE)
    print("Final Accuracy... :", evaluate_loader(model, gumbel_scalar, train_loader, cce_loss, tau))

    all_acc = 0
    count = 0
    all_rewards = list()
    for i_episode in range(NUM_SIMULATIONS):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            bb_action, latent_x = policy(state)  # backbone latent x
            action = torch.argmax(  model(latent_x.view(1, -1).to(DEVICE), gumbel_scalar, tau)[0]  ).item()  # wrapper prediction
            state, reward, done, _ = env.step(action)
            running_reward += reward
            all_acc += bb_action == action
            count += 1
            if done:
                break

            

        data_rewards.append(running_reward)
        print("Running Reward:", running_reward)
        
    data_accuracy.append(all_acc / count)
    print("Reward: ",  sum(data_rewards) / len(data_rewards)) # Average Reward
    print("Accuracy: ", sum(data_accuracy) / len(data_accuracy))
    # log the reward and Acc
    writer.add_scalar("Reward", sum(data_rewards) / len(data_rewards))
    writer.add_scalar("Accuracy", sum(data_accuracy) / len(data_accuracy))
    
    with open(results_file, 'a') as f:
        f.write(f"Reward: {sum(data_rewards) / len(data_rewards)}, Accuracy: {sum(data_accuracy) / len(data_accuracy)}\n")

data_accuracy = np.array(data_accuracy)
data_rewards = np.array(data_rewards)


print(" ")
print("===== Data Accuracy:")
print("Accuracy:", data_accuracy)
print("Mean:", data_accuracy.mean())
print("Standard Error:", data_accuracy.std() / np.sqrt(NUM_ITERATIONS))
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS))


with open(results_file, 'a') as f:
    f.write("\n===== Data Accuracy:\n")
    f.write(f"Accuracy:  {data_accuracy}\n")
    f.write(f"Mean: {data_accuracy.mean()}\n")
    f.write(f"Standard Error: {data_accuracy.std() / np.sqrt(NUM_ITERATIONS)}\n")
    f.write("\n===== Data Reward:\n")
    f.write(f"Rewards:  {data_rewards}\n")
    f.write(f"Mean: {data_rewards.mean()}\n")
    f.write(f"Standard Error: {data_rewards.std() / np.sqrt(NUM_ITERATIONS)}\n")
