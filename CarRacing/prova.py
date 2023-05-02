import pickle
import numpy as np
from PIL import Image
from games.carracing import RacingNet, CarRacing

# 22566

with open('/media/caterina/287CD7D77CD79E3E/cate/X_train_observations.pkl', 'rb') as f:    
    x_train_o = pickle.load(f)

x_train_o = np.array([item for sublist in x_train_o for item in sublist])

# x_train_o: numpy array
print(len(x_train_o)) # 3713

env = CarRacing(frame_skip=0, frame_stack=4,)
#env.render()

#print(type(X_train)) # numpy array

#p_idxs = np.array([0]) 
img = x_train_o[3712]
#print(type(img), type(X_train))  numpy array, numpy array
#print(type(img))

img = Image.fromarray(img, 'RGB')
img.save('/media/caterina/287CD7D77CD79E3E/cate/prototypes/PROT1.png')

#im.show("p1.png")
