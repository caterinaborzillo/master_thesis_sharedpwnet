import gym
from PIL import Image

env = gym.make('CarRacing-v1')
observation = env.reset()

# Loop through the environment
for i in range(3):
    # Render the environment
    #env.render()
    
    # Get the observation image
    #img = Image.fromarray(observation, 'RGB')
    
    img = env.render(mode='rgb_array')
    # type(img): --> numpy array
    
    img = Image.fromarray(img, 'RGB')
    #img.resize(size=(96, 96))
    # Save the image as a .png file
    img.save(f'/media/caterina/obs_{i}.png')
    #img.show(f'obs_{i}.png')
    # Take a random action in the environment
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    # Exit the loop if the episode is done
    if done:
        break























'''
import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import path
from PIL import Image

from matplotlib import cm


with open('data/states.pkl', 'rb') as f:
    X_train = pickle.load(f)

X_train = np.array([item.cpu() for sublist in X_train for item in sublist])

print(type(X_train))
print(len(X_train))

#print(len(X_train))

#print(len(X_train))
p_idxs = np.array([0]) 
img = X_train[p_idxs]


im = Image.fromarray(img.reshape(96, 96, 3), 'RGB')

im.show("p1.png")
'''