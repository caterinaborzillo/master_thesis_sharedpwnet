import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml

from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import gumbel_softmax, cosine_similarity
from argparse import ArgumentParser
import os
from os.path import join
from itertools import combinations
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor

NUM_ITERATIONS = 5
CONFIG_FILE = "config.toml"
#MODEL_DIR = 'weights/myprotonet.pth'
BATCH_SIZE = 32
LATENT_SIZE = 256
NUM_EPOCHS = 100
PROTOTYPE_SIZE = 50
#NUM_PROTOTYPES = 7
NUM_CLASSES = 3
DEVICE = 'cuda'
SIMULATION_EPOCHS = 30
#NUM_SLOTS_PER_CLASS = 2
clst_weight = 0.008 # before: 0.08
sep_weight = -0.0008 # before: 0.008
l1_weight = 1e-5 #1e-4

num_proto = [4, 6] #num_proto = [4, 6, 9]
num_slots = [2] #num_slots = [1 ,2 ,3]

def print_info():
    print("Training INFO:")
    print(f"BATCH SIZE: {BATCH_SIZE}, NUM PROTOTYPES: {NUM_PROTOTYPES}, NUM SLOTS PER CLASS: {NUM_SLOTS_PER_CLASS}")
    print(f"NUM ITERATIONS: {NUM_ITERATIONS}, NUM TRAINING EPOCHS: {NUM_EPOCHS}, SIMULATION EPOCHS: {SIMULATION_EPOCHS}")
    print(f"Loss INFO --> clst_weight: {clst_weight}, sep_weight: {sep_weight}, l1_weight: {l1_weight}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    return

#print_info()

class MyProtoNet(nn.Module):
    def __init__(self):
        super(MyProtoNet, self).__init__()
        self.projection_network = nn.Sequential(
            nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
            nn.InstanceNorm1d(PROTOTYPE_SIZE),
            nn.ReLU(),
            nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
        )
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
        self.relu = nn.ReLU()
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
        out.T[0] = self.tanh(out.T[0]) # steering between -1 and +1
        out.T[1] = self.relu(out.T[1]) # acc > 0
        out.T[2] = self.relu(out.T[2]) # brake > 0 
        return out
    
    def forward(self, x, gumbel_scalar):
        '''
        x (raw input) size: (batch, 256)
        '''
        if gumbel_scalar == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scalar, dim=1, tau = 0.5)
        
        x = self.projection_network(x)
        similarity = self.prototype_layer(x)
        
        mixed_similarity = torch.einsum('bp, cpn->bcn', similarity, proto_presence) # (batch, NUM_CLASSES, NUM_SLOTS_PER_CLASS)

        out1 = self.class_identity_layer(mixed_similarity.flatten(start_dim=1))
        
        out2 = self.output_activations(out1)
        return out2, x, similarity, proto_presence

def evaluate_loader(model, gumbel_scalar, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # size of imgs: [batch, 256], size of labels: [batch, 3]
            logits, _, _, _ = model(imgs, gumbel_scalar)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total

start_val = 1.3
end_val = 10 **3 
epoch_interval = 30 # before: 10
alpha2 = (end_val / start_val) ** 2 / epoch_interval

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

def maximum(a, b, c): 
  
    if (a >= b) and (a >= c): 
        largest = a 
  
    elif (b >= a) and (b >= c): 
        largest = b 
    else: 
        largest = c 
          
    return largest 

if not os.path.exists('results/'):
    os.makedirs('results/')
if not os.path.exists('prototypes/'):
    os.makedirs('prototypes/')
    
for p in num_proto: 
    for s in num_slots: 
        
        NUM_PROTOTYPES = p
        NUM_SLOTS_PER_CLASS = s
        print(f"NUM_PROTOTYPES: {NUM_PROTOTYPES}")
        print(f"NUM_SLOTS: {NUM_SLOTS_PER_CLASS}")
        
        with open('results/myprotonet_results.txt', 'a') as f:
            f.write("--------------------------------------------------------------------------------------------------------------------------\n")
            f.write(f"model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}\n")
            f.write(f"NUM_PROTOTYPES: {NUM_PROTOTYPES}\n")
            f.write(f"NUM_SLOTS: {NUM_SLOTS_PER_CLASS}\n")
        
        #MODEL_DIR = f'weights/model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}'
        
        data_rewards = list()
        data_errors = list()

        for iter in range(NUM_ITERATIONS):
            
            MODEL_DIR = f'weights/model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/'
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            
            MODEL_DIR_ITER = f'weights/model_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/iter_{iter}.pth'
            
            with open('results/myprotonet_results.txt', 'a') as f:
                f.write(f"ITERATION {iter}: \n")
                
            writer = SummaryWriter(f"runs/myprotonet_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}/Iteration_{iter}")
            
            cfg = load_config()
            env = CarRacing(frame_skip=0, frame_stack=4,)
            net = RacingNet(env.observation_space.shape, env.action_space.shape)
            ppo = PPO(
                env,
                net,
                lr=cfg["lr"],
                gamma=cfg["gamma"],
                batch_size=cfg["batch_size"],
                gae_lambda=cfg["gae_lambda"],
                clip=cfg["clip"],
                value_coef=cfg["value_coef"],
                entropy_coef=cfg["entropy_coef"],
                epochs_per_step=cfg["epochs_per_step"],
                num_steps=cfg["num_steps"],
                horizon=cfg["horizon"],
                save_dir=cfg["save_dir"],
                save_interval=cfg["save_interval"],
            )
            # agent weights
            ppo.load("weights/agent_weights.pt")

            with open('data/X_train.pkl', 'rb') as f:
                X_train = pickle.load(f)
            with open('data/real_actions.pkl', 'rb') as f:
                real_actions = pickle.load(f)
            #with open('/media/caterina/287CD7D77CD79E3E/cate/X_train_observations.pkl', 'rb') as f:
            #    X_train_observations = pickle.load(f)
            #X_train_observations = np.array([item for sublist in X_train_observations for item in sublist])
            
            with open('data/obs_train.pkl', 'rb') as f:
                X_train_observations = pickle.load(f)
            X_train_observations = np.array([item for sublist in X_train_observations for item in sublist])

            X_train = np.array([item for sublist in X_train for item in sublist])
            real_actions = np.array([item for sublist in real_actions for item in sublist])
            tensor_x = torch.Tensor(X_train)
            tensor_y = torch.tensor(real_actions, dtype=torch.float32)
            train_dataset = TensorDataset(tensor_x.to(DEVICE), tensor_y.to(DEVICE))
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
            
            #### Train
            model = MyProtoNet().eval()
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
                train_error = evaluate_loader(model, gumbel_scalar, train_loader, mse_loss)
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
                            _, x, _, _ = model(img_tensor.to(DEVICE), gumbel_scalar)
                            # x è lo stato s dopo la projection network
                            transformed_x.append(x[0].tolist())
                    transformed_x = np.array(transformed_x)
                    
                    list_projected_prototype = list()
                    for i in range(NUM_PROTOTYPES):
                        # I take the trained prototype p
                        trained_p = model.projection_network(model.prototypes)
                        trained_prototype_clone = trained_p.clone().detach()[i].view(1,-1)
                        trained_prototype = trained_prototype_clone.cpu()
                        # apply knn to find the top 4 x states which are the closest to the 4 prototypes
                        knn = KNeighborsRegressor(algorithm='brute')
                        knn.fit(transformed_x, list(range(len(transformed_x)))) # lista da 0 a len(transformed_x) - n of training data
                        #dist: distance (minimum one) , transf_idx: index of the transformed state closer to the prototype p
                        dist, transf_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
                        #print(f"Trained prototype p{i}:")
                        #print(f"distance: {dist.item()}, index of nearest point: {transf_idx.item()}")
                        projected_prototype = X_train[transf_idx.item()]# transformed_x[transf_idx.item()]
                        list_projected_prototype.append(projected_prototype.tolist())
                        
                        if epoch == NUM_EPOCHS-20-2: 
                            print("I'm saving prototypes' images in prototypes/ directory...")
                            prototype_image = X_train_observations[transf_idx.item()]
                            prototype_image = Image.fromarray(prototype_image, 'RGB')
                            #prototype_image.save(f'/media/caterina/287CD7D77CD79E3E/cate/prototypes/prototype_{i+1}.png')
                            prototype_image.save(f'prototypes/prototype{i+1}_myprotonet_p{NUM_PROTOTYPES}_s{NUM_SLOTS_PER_CLASS}_iter{iter}.png')
                        
                    trained_prototypes = model.prototypes.clone().detach()
                    # praticamente vado a sostituire i prototipi allenati durante il training con gli stati (dopo la projection_network) che sono più vicini ai prototipi
                    # è come se facessi una proiezione dei prototipi (allenati da zero) sugli stati (veri stati nel training set)
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
                                
                    '''
                    for name, param in model.named_parameters():
                        if first_time:
                            if param.requires_grad:
                                print("--- ", name, "---> training")
                            else:
                                print("--- ", name, "---> freezed")
                    first_time=False
                    '''
                
                for instances, labels in train_loader:
                    optimizer.zero_grad()
                            
                    instances, labels = instances.to(DEVICE), labels.to(DEVICE)
                    logits, _, similarity, proto_presence = model(instances, gumbel_scalar)
                    
                    loss1 = mse_loss(logits, labels) 
                    # orthogonal loss --> for slots orthogonality: in this way successive slots of a class are assigned to different prototypes
                    orthogonal_loss = torch.Tensor([0]).to(DEVICE)
                    '''
                    for c in range(0, model.proto_presence.shape[0], 1000): # model.proto_presence.shape[0]: NUM_CLASSES
                        orthogonal_loss_p = cosine_similarity(model.proto_presence.unsqueeze(2)[c:c+1000],model.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                        orthogonal_loss += orthogonal_loss_p
                    orthogonal_loss = orthogonal_loss / (NUM_SLOTS_PER_CLASS * NUM_CLASSES) - 1 # page 7 of paper
                    '''
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
                        #label = [sterring between -1 and +1, accelerating >0, braking >0]
                        max_value = maximum(abs(label[0]), label[1], label[2])
                        if max_value == abs(label[0]):
                            labels_pp.append(0)  
                        elif max_value == label[1]:
                            labels_pp.append(1)
                        else:
                            labels_pp.append(2)
                    
                    proto_presence = proto_presence[labels_pp] # (?) labels_pp deve essere un vettore (batch size, classe) classe = 0,1,2
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
            
                print("Epoch:", epoch, "Loss:", running_loss / len(train_loader), "Train_error:", train_error)
                with open('results/myprotonet_results.txt', 'a') as f:
                    f.write(f"Epoch: {epoch}, Loss: {running_loss / len(train_loader)}, Train_error: {train_error}\n")
                #writer.add_scalar("Loss_mse/train", running_loss_mse/len(train_loader), epoch)
                #writer.add_scalar("Loss_clst/train", running_loss_clst/len(train_loader), epoch)
                #writer.add_scalar("Loss_sep/train", running_loss_sep/len(train_loader), epoch)
                #writer.add_scalar("Loss_l1/train", running_loss_l1/len(train_loader), epoch)
                #writer.add_scalar("Loss_ortho/train", running_loss_ortho/len(train_loader), epoch)
                writer.add_scalar("Running_loss", running_loss/len(train_loader), epoch)
                writer.add_scalar("Train_error", train_error, epoch)
                running_loss = running_loss_mse = running_loss_clst = running_loss_sep = running_loss_l1 =  running_loss_ortho = 0.
                    
                scheduler.step()
            
            states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []
            self_state = ppo._to_tensor(env.reset())

            # Wrapper model with learned weights
            #model.eval()
            
            # Wrapper model with learned weights
            model = MyProtoNet().eval()
            model.load_state_dict(torch.load(MODEL_DIR_ITER))
            model.to(DEVICE)
            print("Checking for the error... :", evaluate_loader(model, gumbel_scalar, train_loader, mse_loss))
    
            reward_arr = []
            all_errors = list()
            for i in tqdm(range(SIMULATION_EPOCHS)):
                state = ppo._to_tensor(env.reset())
                count = 0
                rew = 0
                rew_list = []
                model.eval()

                for t in range(10000):
                    # Get black box action
                    # value network: estimates the value of being in the state "state"
                    value, alpha, beta, latent_x = ppo.net(state)
                    value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
                    # policy network: uses estimates of the value function to select actions that are more likely to lead higher rewards
                    policy = Beta(alpha, beta)
                    input_action = policy.mean.detach()
                    bb_action = ppo.env.preprocess(input_action.cpu().numpy())
                    # perform the action "input_action" found by the policy network
                    ##_, _, _, _, bb_action = ppo.env.step(input_action.cpu().numpy())
                    # bb_action size [3] --> è l'azione della stessa forma dell'output del mio modello (per poterli confrontare)
                    action, _, _, _ = model(latent_x.to(DEVICE), gumbel_scalar)
                    # action size [1,3]
                    all_errors.append(mse_loss(torch.tensor(bb_action).to(DEVICE), action[0]).detach().item())

                    state, reward, done, _, _ = ppo.env.step(action[0].detach().cpu().numpy(), real_action=True)
                    state = ppo._to_tensor(state)
                    rew += reward
                    rew_list.append(reward)
                    count += 1
                    
                    if done:
                        break
                reward_arr.append(rew)

            data_rewards.append(sum(reward_arr) / SIMULATION_EPOCHS)
            data_errors.append(sum(all_errors) / SIMULATION_EPOCHS)
            print("Data reward: ", sum(reward_arr) / SIMULATION_EPOCHS)
            print("Data error: ", sum(all_errors) / SIMULATION_EPOCHS)
            # log the reward and MAE
            writer.add_scalar("Reward", sum(reward_arr) / SIMULATION_EPOCHS, iter)
            writer.add_scalar("MAE", sum(all_errors) / SIMULATION_EPOCHS, iter)
            
            with open('results/myprotonet_results.txt', 'a') as f:
                f.write(f"Data reward: {sum(reward_arr) / SIMULATION_EPOCHS}, Data error: {sum(all_errors) / SIMULATION_EPOCHS}\n")

        data_errors = np.array(data_errors)
        data_rewards = np.array(data_rewards)


        print(" ")
        print("===== Data MAE:")
        print("Errors:", data_errors)
        print("Mean:", data_errors.mean())
        print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS))
        print(" ")
        print("===== Data Reward:")
        print("Rewards:", data_rewards)
        print("Mean:", data_rewards.mean())
        print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS))
        
        
        with open('results/myprotonet_results.txt', 'a') as f:
            f.write("\n===== Data MAE:\n")
            f.write(f"Mean: {data_errors.mean()}\n")
            f.write(f"Standard Error: {data_errors.std() / np.sqrt(NUM_ITERATIONS)}\n")
            f.write("\n===== Data Reward:\n")
            f.write(f"Rewards:  {data_rewards}\n")
            f.write(f"Mean: {data_rewards.mean()}\n")
            f.write(f"Standard Error: {data_rewards.std() / np.sqrt(NUM_ITERATIONS)}\n")
            
            
            
