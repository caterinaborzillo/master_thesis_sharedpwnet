import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml
import argparse

from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from PIL import Image
import datetime
import random

from TD3 import TD3

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import gumbel_softmax, cosine_similarity
from argparse import ArgumentParser
import os
from os.path import join
from itertools import combinations
from torch.distributions import Beta
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()

parser.add_argument("n_proto", type=int, default = 8, help="Number of prototypes to be learned")
parser.add_argument("n_slots", type=int, default = 2, help="Number of slots per class")
parser.add_argument("new_proto_init", nargs='?', default=False, const=True, type=bool, help='Specify new proto initialization argument')

args = parser.parse_args()

NUM_PROTOTYPES = args.n_proto
NUM_SLOTS_PER_CLASS = args.n_slots

NUM_ITERATIONS = 15
NUM_EPOCHS = 100
NUM_CLASSES = 4

CONFIG_FILE = "config.toml"
BATCH_SIZE = 128
LATENT_SIZE = 300
PROTOTYPE_SIZE = 50
DEVICE = 'cuda'
SIMULATION_EPOCHS = 30 

name_file = "run_sharedpwnet"

current_date = datetime.date.today()
date = current_date.strftime("%d_%m_%Y")

clst_weight = 0.008 # better than 0.08
sep_weight = -0.0008 # better than 0.008
l1_weight = 1e-5 # better than 1e-4

env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 30
lr = 0.002
max_timesteps = 2000
render = True
save_gif = False
#filename = "TD3_{}_{}".format(env_name, random_seed)
#filename += '_solved'
filename = "TD3_BipedalWalker-v2_0_solved"
#directory = "./preTrained/{}".format(env_name)
directory = "./preTrained/BipedalWalker-v2/ONE"
env = gym.make(env_name, hardcore=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(lr, state_dim, action_dim, max_action)
policy.load_actor(directory, filename)

# novel initialization
X_train = np.load('data/X_train.npy')
a_train = np.load('data/a_train.npy')

def normalize_list(values):
    min_value = min(values)
    max_value = max(values)
    
    normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
    return normalized_values

list_actions = {}
medians = {}
for actions in a_train:
    for action_id in range(NUM_CLASSES):
        list_actions[action_id] = []
        a = actions[action_id]
        list_actions[action_id].append(a)
        
for action_id in range(NUM_CLASSES):        
    m = np.median(list_actions[action_id])
    medians[action_id] = m

states_actions_zip = []
for state, action in zip(X_train, a_train):
	states_actions_zip.append((state, action))

state_actions = {}
for action_id in range(NUM_CLASSES): 						
	state_actions[action_id] = []						

	action_values = [x[1][action_id] for x in states_actions_zip]   

	median = medians[action_id]
        
	diff = [np.abs(p-median) for p in action_values]
	normalized_diff = normalize_list(diff)

	probs_50_perc = np.percentile(normalized_diff, 50)			

	for (state, action), action_probs in zip(states_actions_zip, normalized_diff):
		if action_probs > probs_50_perc:
			state_actions[action_id].append(state)		# state_actions = {0: [stato1>50,stato2>50, stato3>50, ...]  1: , 2: }

prototypes = []

for action_id in range(NUM_CLASSES):
	prototypes.append(KMeans(NUM_SLOTS_PER_CLASS, n_init="auto").fit(state_actions[action_id]).cluster_centers_)   # prototypes = [[p11,p12,p13],[p21,p22,p23],[p31,p32,p33],]

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
            
        self.class_identity_layer = nn.Linear(NUM_SLOTS_PER_CLASS * NUM_CLASSES, NUM_CLASSES, bias=False) 
        positive_one_weights_locations = torch.t(self.prototype_class_identity) # transpose
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = 0 # -0.5
        # to weight in the proper way the last linear layer
        self.class_identity_layer.weight.data.copy_(correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations)
        
        self.tanh = nn.Tanh()
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
        return self.tanh(out)
    
    def forward(self, x, gumbel_scalar, tau):
        '''
        x (raw input) size: (batch, 24) 
        '''
        if gumbel_scalar == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scalar, dim=1, tau=tau)
        
        x = self.projection_network(x)
        similarity = self.prototype_layer(x)
        
        mixed_similarity = torch.einsum('bp, cpn->bcn', similarity, proto_presence) # (batch, NUM_CLASSES, NUM_SLOTS_PER_CLASS)

        out1 = self.class_identity_layer(mixed_similarity.flatten(start_dim=1))
        
        out2 = self.output_activations(out1)
        return out2, x, similarity, proto_presence

def evaluate_loader(model, gumbel_scalar, loader, loss, tau):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # size of imgs: [batch, 256], size of labels: [batch, 3]
            logits, _, _, _ = model(imgs, gumbel_scalar, tau)
            
            current_loss = loss(logits, labels)
            total_loss += current_loss.item()
            total += len(logits)
    model.train()
    return total_loss / len(loader)

start_val = 1.3
end_val = 10 **3 
epoch_interval = 30 # or 10
alpha2 = (end_val / start_val) ** 2 / epoch_interval
alpha3 = 3.4 * 10**4

def lambda1(epoch): return start_val * np.sqrt((alpha2 * (epoch))) if epoch < epoch_interval else end_val

def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config

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

def maximum(a, b, c, d): 
  
    if (a >= b) and (a >= c) and (a >= d): 
        largest = a 
  
    elif (b >= a) and (b >= c) and (b >= d): 
        largest = b 
    
    elif (c >= a) and (c >= b) and (c >= d): 
        largest = c 
        
    else: 
        largest = d 
          
    return largest 
        
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
data_errors = list()

for iter in range(NUM_ITERATIONS):
    
    MODEL_DIR = f'weights/{date}_{name_file}_model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if args.new_proto_init:
        prototype_path = f'prototypes/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}_newinit/iter_{iter}/'
    else:
        prototype_path = f'prototypes/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/iter_{iter}/'
    if not os.path.exists(prototype_path):
        os.makedirs(prototype_path)
    
    MODEL_DIR_ITER = f'weights/{date}_{name_file}_model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/iter_{iter}.pth'
    
    with open(results_file, 'a') as f:
        f.write(f"ITERATION {iter}: \n")
        
    writer = SummaryWriter(f"runs/{date}_{name_file}_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/Iteration_{iter}")

    # TO SAVE PROTOTYPES
    obs_train = np.load('data/obs_train.npy')
    
    X_train = np.load('data/X_train.npy')
    a_train = np.load('data/a_train.npy')
    
    tensor_x = torch.Tensor(X_train)
    #print("tensor x size: ", tensor_x.size())
    tensor_y = torch.tensor(a_train, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    
    #### Train
    model = SharedPwNet().eval()
    model.to(DEVICE)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_error = float('inf')
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
        
        train_error = evaluate_loader(model, gumbel_scalar, train_loader, mse_loss, tau)
        model.train()

        if train_error < best_error and epoch > NUM_EPOCHS-20:
            torch.save(model.state_dict(), MODEL_DIR_ITER) # saves model parameters
            best_error = train_error
        
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
                knn.fit(transformed_x, list(range(len(transformed_x)))) 
                dist, transf_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
                projected_prototype = X_train[transf_idx.item()] # transformed_x[transf_idx.item()]
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
        
                
            loss1 = mse_loss(logits, labels) 

                
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
            
            labels_p = labels.cpu().numpy().tolist()
            labels_pp = list()
            for label in (labels_p):
                #label = 4 continuous actions: hip torque, knee torque for each leg each from -1.0 to 1.0
                max_value = maximum(abs(label[0]), abs(label[1]), abs(label[2]), abs(label[3]))
                if max_value == abs(label[0]):
                    labels_pp.append(0)  
                elif max_value == abs(label[1]):
                    labels_pp.append(1)
                elif max_value == abs(label[2]):
                    labels_pp.append(2)
                else:
                    labels_pp.append(3)
                                    
            proto_presence = proto_presence[labels_pp] 
            inverted_proto_presence = 1 - proto_presence
            labels.to(DEVICE)
                
            clst_loss_val = dist_loss(model, similarity, proto_presence, NUM_SLOTS_PER_CLASS)  
            sep_loss_val = dist_loss(model, similarity, inverted_proto_presence, NUM_PROTOTYPES - NUM_SLOTS_PER_CLASS) 
            
            # to remove
            #prototypes_of_correct_class = proto_presence.sum(dim=-1).detach() # dovrebbe essere size: [num class, num prot]
            #prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            #avg_separation_cost = torch.sum(similarity * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,dim=1)
            #avg_separation_cost = torch.mean(avg_separation_cost)
            
            l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
            l1 = (model.class_identity_layer.weight * l1_mask).norm(p=1)
            # We use the following weighting schema for loss function: L entropy = 1.0, L clst = 0.8, L sep = −0.08, L orth = 1.0, and L l 1 = 10 −4 . Finally, 
            # we normalize L orth , dividing it by the number of classes multiplied by the number of slots per class. (page 20)
            loss = loss1 + clst_loss_val * clst_weight + sep_loss_val * sep_weight + l1 * l1_weight + orthogonal_loss 
            
            running_loss_mse += loss1.item()
            running_loss_clst += clst_loss_val.item() * clst_weight
            running_loss_sep += sep_loss_val.item() * sep_weight
            running_loss_l1 += l1.item() * l1_weight
            running_loss_ortho += orthogonal_loss.item() 
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
    
        print("Epoch:", epoch, "Running Loss:", running_loss / len(train_loader), "Train error:", train_error)
        with open(results_file, 'a') as f:
            f.write(f"Epoch: {epoch}, Running Loss: {running_loss / len(train_loader)}, Train error: {train_error}\n")

        writer.add_scalar("Running_loss", running_loss/len(train_loader), epoch)
        writer.add_scalar("Train_error", train_error, epoch)
        running_loss = running_loss_mse = running_loss_clst = running_loss_sep = running_loss_l1 =  running_loss_ortho = 0.
            
        scheduler.step()
    
    #states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []


    # Wrapper model with learned weights
    model = SharedPwNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR_ITER))
    model.to(DEVICE)
    print("Checking for the error... :", evaluate_loader(model, gumbel_scalar, train_loader, mse_loss, tau))

    total_reward = list()
    all_errors = list()
    model.eval()
    for ep in tqdm(range(SIMULATION_EPOCHS)):
        ep_reward = 0
        ep_errors = 0
        state = env.reset()

        for t in range(max_timesteps):
            bb_action, x = policy.select_action(state)
            A, _, _, _ = model( torch.tensor(x, dtype=torch.float32).view(1, -1).to(DEVICE), gumbel_scalar, tau )
            state, reward, done, _ = env.step(A.detach().cpu().numpy()[0])

            ep_reward += reward
            ep_errors += mse_loss( torch.tensor(bb_action).to(DEVICE), A[0]).detach().item()

            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append( ep_reward )
        all_errors.append( ep_errors )
        ep_reward = 0
    
    env.close()

    data_rewards.append(sum(total_reward) / SIMULATION_EPOCHS)
    data_errors.append(sum(all_errors) / SIMULATION_EPOCHS)
    print("Reward: ", sum(total_reward) / SIMULATION_EPOCHS)
    print("MSE: ", sum(all_errors) / SIMULATION_EPOCHS)
    # log the reward and MAE
    writer.add_scalar("Reward", sum(total_reward) / SIMULATION_EPOCHS, iter)
    writer.add_scalar("MSE", sum(all_errors) / SIMULATION_EPOCHS, iter)
    
    with open(results_file, 'a') as f:
        f.write(f"Reward: {sum(total_reward) / SIMULATION_EPOCHS}, MSE: {sum(all_errors) / SIMULATION_EPOCHS}\n")

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)

print(" ")
print("===== Data MAE:")
print("MSE:", data_errors)
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS))
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS))


with open(results_file, 'a') as f:
    f.write("\n===== Data MAE:\n")
    f.write(f"MSE:  {data_errors}\n")
    f.write(f"Mean: {data_errors.mean()}\n")
    f.write(f"Standard Error: {data_errors.std() / np.sqrt(NUM_ITERATIONS)}\n")
    f.write("\n===== Data Reward:\n")
    f.write(f"Rewards:  {data_rewards}\n")
    f.write(f"Mean: {data_rewards.mean()}\n")
    f.write(f"Standard Error: {data_rewards.std() / np.sqrt(NUM_ITERATIONS)}\n")
            
            
            

