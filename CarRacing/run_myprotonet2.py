import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor

NUM_ITERATIONS = 6
CONFIG_FILE = "config.toml"
MODEL_DIR = 'weights2/pwnet_star.pth'
BATCH_SIZE = 32
LATENT_SIZE = 256
NUM_EPOCHS = 50
PROTOTYPE_SIZE = 50
NUM_PROTOTYPES = 4
NUM_CLASSES = 3
DEVICE = 'cuda'
SIMULATION_EPOCHS = 30

class MyProtoNet2(nn.Module):
    def __init__(self):
        super(MyProtoNet2, self).__init__()
        self.proj_nets = ListModule(self, '_proj_nets')
        for i in range(NUM_PROTOTYPES):           
            projection_network = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.InstanceNorm1d(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.proj_nets.append(projection_network)
        self.prototypes = nn.Parameter(torch.randn((NUM_PROTOTYPES, LATENT_SIZE), dtype=torch.float32), requires_grad=True)
        self.final_linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.epsilon = 1e-5
    
    def prototype_layer(self, x, p):
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)            
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        # similarity function from Chen et al. 2019: to score the distance between state c and prototype p
        similarity = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return similarity
    
    def output_activations(self, out):
        out.T[0] = self.tanh(out.T[0]) # steering between -1 and +1
        out.T[1] = self.relu(out.T[1]) # acc > 0
        out.T[2] = self.relu(out.T[2]) # brake > 0 
        return out
    
    def forward(self, x):
        # projection networks through prototypes 
        transf_prototypes = list()
        for i, t in enumerate(self.proj_nets):
            transf_prototypes.append(t(self.prototypes[i].view(1, -1)))
        latent_protos = torch.cat(transf_prototypes, dim=0) 
        
        # projection networks through state x
        similarity = list()
        for i, t in enumerate(self.proj_nets):
            action_prototype = latent_protos[i]
            similarity.append(self.prototype_layer(t(x), action_prototype).view(-1,1))
        similarity = torch.cat(similarity, axis=1)
        
        
        
        after_linear = self.final_linear(similarity)
        out = self.output_activations(after_linear)
        return out, x

def evaluate_loader(model, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # size of imgs: [batch, 256], size of labels: [batch, 3]
            logits, _ = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total

def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config

class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

def clust_loss(x, y, model, criterion):
    """
    Forces each prototype to be close to training data
    """
    ps = model.prototypes  # take prototypes in new feature space
    model = model.eval()
    x = model.projection_network(x)  # transform into new feature space
    b_size = x.shape[0]
    for idx, p in enumerate(ps):
        target = p.repeat(b_size, 1)
        if idx == 0:
            loss = criterion(x, target) 
        else:
            loss += criterion(x, target)
    model = model.train()  
    return loss

def sep_loss(x, y, model, criterion):
    """
    Force each prototype to be far from each other
    """
    
    p = model.prototypes  # take prototypes in new feature space
    model = model.eval()
    x = model.projection_network(x)  # transform into new feature space
    loss = torch.cdist(p, p).sum() / ((NUM_PROTOTYPES**2 - NUM_PROTOTYPES) / 2)
    return -loss 

data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
    
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
    ppo.load("weights2/agent_weights.pt")

    with open('data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/real_actions.pkl', 'rb') as f:
        real_actions = pickle.load(f)

    X_train = np.array([item for sublist in X_train for item in sublist])
    real_actions = np.array([item for sublist in real_actions for item in sublist])
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(real_actions, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x.to(DEVICE), tensor_y.to(DEVICE))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    
    #### Train
    model = MyProtoNet2().eval()
    model.to(DEVICE)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_acc = 0.
    model.train()
    
    #print("Trainable: \n")
    #for name, param in model.named_parameters():
        #if param.requires_grad:
            #print(name, param.data)
            #print("\n")
        
    lambda1 = 1.0
    lambda2 = 0.8
    lambda3 = 0.08
    
    for epoch in range(NUM_EPOCHS):
        model.eval()
        current_acc = evaluate_loader(model, train_loader, mse_loss)
        model.train()
        
        if current_acc > best_acc:
            torch.save(model.state_dict(), MODEL_DIR)
            best_acc = current_acc
        
        for instances, labels in train_loader:
            optimizer.zero_grad()
                    
            instances, labels = instances.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(instances)
                    
            loss1 = mse_loss(logits, labels) * lambda1
            #loss2 = clust_loss(instances, labels, model, mse_loss) * lambda2
            #loss3 = sep_loss(instances, labels, model, mse_loss) * lambda3
            #loss  = loss1 + loss2 + loss3    
            loss1.backward()
            optimizer.step()
            
        scheduler.step()

    # Project Prototypes
    model.eval()
    model.load_state_dict(torch.load(MODEL_DIR))
    print("Accuracy Before Projection:", evaluate_loader(model, train_loader, mse_loss))
    trans_x = list()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(X_train))):
            img = X_train[i]
            # x è lo stato s dopo il f_end (projection network)
            img_tensor = torch.tensor(img, dtype=torch.float32).view(1, -1)
            _, x = model(img_tensor.to(DEVICE))
            trans_x.append(x[0].tolist())
    trans_x = np.array(trans_x)

    nn_xs = list()
    nn_as = list()
    nn_human_images = list()
    for i in range(NUM_PROTOTYPES):
        trained_prototype_clone = model.prototypes.clone().detach()[i].view(1,-1)
        trained_prototype = trained_prototype_clone.cpu()
        knn = KNeighborsRegressor(algorithm='brute')
        knn.fit(trans_x, list(range(len(trans_x)))) # lista da 0 a len(tran_x)
        dist, nn_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
        print(f"Trained prototype p{i}: \n")
        print(f"distance: {dist.item()}, index of nearest point: {nn_idx.item()} \n")
        nn_x = trans_x[nn_idx.item()]    
        nn_xs.append(nn_x.tolist())
    trained_prototypes = model.prototypes.clone().detach()
    # praticamente vado a sostituire i prototipi allenati durante il training con gli stati (dopo f_enc/projection_network) che sono più vicini ai prototipi
    # è come se facessi una proiezione dei prototipi (allenati da zero) sugli stati (veri stati nel training set)
    nn_xs_tensor = torch.tensor(nn_xs, dtype=torch.float32)
    model.prototypes = torch.nn.Parameter(nn_xs_tensor.to(DEVICE))
    torch.save(model.state_dict(), MODEL_DIR)
    print("Accuracy After Projection:", evaluate_loader(model, train_loader, mse_loss))
    
    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []
    self_state = ppo._to_tensor(env.reset())

    # Wapper model with learned weights
    model.eval()
    reward_arr = []
    all_errors = list()
    for i in tqdm(range(SIMULATION_EPOCHS)):
        state = ppo._to_tensor(env.reset())
        count = 0
        rew = 0
        model.eval()

        for t in range(10000):
            # Get black box action
            value, alpha, beta, latent_x = ppo.net(state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
            policy = Beta(alpha, beta)
            input_action = policy.mean.detach()
            _, _, _, _, bb_action = ppo.env.step(input_action.cpu().numpy())

            action = model(latent_x.to(DEVICE))
            all_errors.append(  mse_loss(bb_action.to(DEVICE), action[0][0]).detach().item()  )

            state, reward, done, _, _ = ppo.env.step(action[0][0].detach().cpu().numpy(), real_action=True)
            state = ppo._to_tensor(state)
            rew += reward
            count += 1
            
            if done:
                break

        reward_arr.append(rew)

    data_rewards.append(  sum(reward_arr) / SIMULATION_EPOCHS  )
    data_errors.append(  sum(all_errors) / SIMULATION_EPOCHS )

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)


print(" ")
print("===== Data MAE:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )