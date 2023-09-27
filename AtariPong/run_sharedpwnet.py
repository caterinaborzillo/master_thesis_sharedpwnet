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
import datetime

from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument("n_proto", type=int, default = 6, help="Number of prototypes to be learned")
parser.add_argument("n_slots", type=int, default = 2, help="Number of slots per class")
parser.add_argument("new_proto_init", nargs='?', default=False, const=True, type=bool, help='Specify new proto initialization argument')

args = parser.parse_args()

NUM_PROTOTYPES = args.n_proto
NUM_SLOTS_PER_CLASS = args.n_slots

NUM_ITERATIONS = 15
NUM_EPOCHS = 100
NUM_CLASSES = 6

LATENT_SIZE = 1536
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
delay_ms = 0
SIMULATION_EPOCHS = 30

ENVIRONMENT = "PongDeterministic-v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODELS = False  # Save models to file so you can test later
MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = False  # Train model while playing (Make it False when testing a model)
LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 900  # Load Xth episode from file
BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode
NUM_EPISODES = 3
MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train
GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.99  # Epsilon decay rate by step

clst_weight = 0.008 # before: 0.08
sep_weight = -0.0008 # before: 0.008
l1_weight = 1e-5 #1e-4


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """

    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q, x

class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition for Agent
        """

        # State size for breakout env. SS images (210, 160, 3). Used as input size in network
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Activation size for breakout env. Used as output size in network
        self.action_size = environment.action_space.n

        # Image pre process params
        self.target_h = 80  # Height after process
        self.target_w = 64  # Widht after process

        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table

        # Trust rate to our experiences
        self.gamma = GAMMA  # Discount coef for future predictions
        self.alpha = ALPHA  # Learning Rate

        # After many experinces epsilon will be 0.05
        # So we will do less Explore more Exploit
        self.epsilon = 0  # Explore or Exploit
        self.epsilon_decay = EPSILON_DECAY  # Adaptive Epsilon Decay Rate
        self.epsilon_minimum = 0.05  # Minimum for Explore

        # Deque holds replay mem.
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        # Create two model for DDQN algorithm
        self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # Adam used as optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def act(self, state):
        """
        Get state and do action
        Two option can be selectedd if explore select random action
        if exploit ask nnet for action
        """

        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
            state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
            q_values, x = self.online_model.forward(state)  # (1, action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values, x = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

        return action, x

    def train(self):
        """
        Train neural nets with replay memory
        returns loss and max_q val predicted from online_net
        """
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        # We get out minibatch and turn it to numpy array
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        # Concat batches in one array
        # (np.arr, np.arr) ==> np.BIGarr
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every step
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay

name_file = "run_sharedpwnet"

current_date = datetime.date.today()
date = current_date.strftime("%d_%m_%Y")

with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/a_train.pkl', 'rb') as f:
    a_train = pickle.load(f)

with open('data/obs_train.pkl', 'rb') as f:
    X_train_observations = pickle.load(f)

'''
def normalize_list(values):
    min_value = min(values)
    max_value = max(values)
    
    normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
    return normalized_values
'''
# ogni azione non è come in Carracing [-0.584, 0.0, 0.456] ma ogni azioni in Ataripong è un intero [4]
# actions space= [0,1,2,3,4,5]

# qui mappo stato e azione corrispondente del dataset 
action_states = {}
for action_id in range(NUM_CLASSES):
    action_states[action_id] = []
    
    for state, action in zip(X_train, a_train):
        if action == action_id:
            action_states[action_id].append(state)
        
'''state_actions = {}
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
'''

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
    
    environment = gym.make(ENVIRONMENT) # , render_mode='human')  # Get env
    environment.seed(0)
    agent = Agent(environment)  # Create Agent
    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(MODEL_PATH+str(LOAD_FILE_EPISODE)+".pkl", map_location=torch.device('cpu')))
        with open(MODEL_PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')
        startEpisode = LOAD_FILE_EPISODE + 1
    else:
        startEpisode = 1
    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
    total_step = 1  # Cumulkative sum of all steps in episodes

    
    X_train = np.array(X_train)
    a_train = np.array(a_train)
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.long)
    train_dataset = TensorDataset(tensor_x.to(DEVICE), tensor_y.to(DEVICE))
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
                    prototype_image = X_train_observations[transf_idx.item()]
                    for j, frame in enumerate(prototype_image):
                        prototype_image = Image.fromarray(frame, 'RGB')
                        p_path = prototype_path+f'p{i+1}_'+f'FRAME{j+1}.png'
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
            # We use the following weighting schema for loss function: L entropy = 1.0, L clst = 0.8, L sep = −0.08, L orth = 1.0, and L l 1 = 10 −4 . Finally, 
            # we normalize L orth , dividing it by the number of classes multiplied by the number of slots per class. (page 20)
            loss = loss1 + clst_loss_val * clst_weight + sep_loss_val * sep_weight + l1 * l1_weight + orthogonal_loss 
            #print(loss1, clst_loss_val * clst_weight, sep_loss_val * sep_weight, l1 * l1_weight , orthogonal_loss )
            #loss2 = clust_loss(instances, labels, model, mse_loss) * lambda22
            #loss3 = sep_loss(instances, labels, model, mse_loss) * lambda33
            
            running_loss_mse += loss1.item()
            running_loss_clst += clst_loss_val.item() * clst_weight
            running_loss_sep += sep_loss_val.item() * sep_weight
            running_loss_l1 += l1.item() * l1_weight
            running_loss_ortho += orthogonal_loss.item() 
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
    
        print("Epoch:", epoch, "Running Loss:", running_loss / len(train_loader), "Current accuracy:", current_acc)
        with open(results_file, 'a') as f:
            f.write(f"Epoch: {epoch}, Running Loss: {running_loss / len(train_loader)}, Current accuracy: {current_acc}\n")
        #writer.add_scalar("Loss_mse/train", running_loss_mse/len(train_loader), epoch)
        #writer.add_scalar("Loss_clst/train", running_loss_clst/len(train_loader), epoch)
        #writer.add_scalar("Loss_sep/train", running_loss_sep/len(train_loader), epoch)
        #writer.add_scalar("Loss_l1/train", running_loss_l1/len(train_loader), epoch)
        #writer.add_scalar("Loss_ortho/train", running_loss_ortho/len(train_loader), epoch)
        writer.add_scalar("Running_loss: ", running_loss/len(train_loader), epoch)
        writer.add_scalar("Current_accuracy: ", current_acc, epoch)
        running_loss = running_loss_mse = running_loss_clst = running_loss_sep = running_loss_l1 =  running_loss_ortho = 0.
            
        scheduler.step()
    
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    
    # Wrapper model with learned weights
    model = SharedPwNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR_ITER))
    model.to(DEVICE)
    print("Final accuracy... :", evaluate_loader(model, gumbel_scalar, train_loader, cce_loss, tau))

    all_rewards = list()
    all_acc = list()
    for episode in tqdm(range(SIMULATION_EPOCHS)):
        startTime = time.time()  # Keep time
        state = environment.reset()  # Reset env

        state = agent.preProcess(state)  # Process image
        
        # Stack state . Every state contains 4 time contionusly frames
        # We stack frames like 4 channel image
        state = np.stack((state, state, state, state))
        
        total_max_q_val = 0  # Total max q vals
        total_reward = 0     # Total reward for each episode
        total_loss = 0       # Total loss for each episode
        total_acc = list()
        model.eval()

        for step in range(MAX_STEP):
            
            # Select and perform an action
            agent_action, latent_x = agent.act(state)  # Act
            action, _, _, _ = model(latent_x.to(DEVICE), gumbel_scalar, tau)
            action = torch.argmax(action).item()

            # print(agent_action, action)

            # Normally the randomness is the number on the right (.049...)
            # But as PW-Net is trained on the data from the original model which was already random
            # we lower the randomness here for a fairer comparison.
            # PW-Net here is trained on ~5% random data, plus 0.025 randomness
            if np.random.random_sample() < .025:   #  .04953625663766238:
                action = np.random.randint(0, 5)

            next_state, reward, done, info = environment.step(action)  # Observe

            next_state = agent.preProcess(next_state)  # Process image

            # Stack state . Every state contains 4 time contionusly frames
            # We stack frames like 4 channel image
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            # Store the transition in memory
            agent.storeResults(state, action, reward, next_state, done)  # Store to mem

            # Move to the next state
            state = next_state  # Update state

            total_reward += reward
            total_acc.append( agent_action == action )

            if done:
                all_rewards.append(total_reward)
                all_acc.append( sum(total_acc) / len(total_acc ) )
                break
            

    data_rewards.append(sum(all_rewards) / SIMULATION_EPOCHS)
    data_accuracy.append(sum(all_acc) / SIMULATION_EPOCHS)
    print("Reward: ", sum(all_rewards) / SIMULATION_EPOCHS)
    print("Accuracy: ", sum(all_acc) / SIMULATION_EPOCHS)
    
    writer.add_scalar("Reward", sum(all_rewards) / SIMULATION_EPOCHS, iter)
    writer.add_scalar("Accuracy", sum(all_acc) / SIMULATION_EPOCHS, iter)
    
    with open(results_file, 'a') as f:
        f.write(f"Reward: {sum(all_rewards) / SIMULATION_EPOCHS}, Accuracy: {sum(all_acc) / SIMULATION_EPOCHS}\n")

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
