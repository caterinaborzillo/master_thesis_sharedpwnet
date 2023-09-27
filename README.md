# Understanding Deep RL agent decisions: a novel interpretable approach with trainable prototypes (Shared-PW-Net)


## To reproduce the CODE:
### Environments: Car Racing, Atari Pong, Bipedal Walker, Lunar Lander

- First place inside the environment directory (CarRacing/AtariPong/BipedalWalker/LunarLander) on which you want to conduct experiments (ex. CarRacing). 
```
cd CarRacing
```

- Then run this code to collect the data needed to train the interpretable neural networks for the chosen environment:
```
python collect_data.py
```

Then, in order to train the networks (all networks are trained for 100 epochs), always inside the environment directory, run:

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet_star.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet_star_star.py
```

- For the proposed approach - Shared-PW-Net - you have to specify:
      - the number of prototypes that the network has to learn
      - the number of slots per class
      - if you want to apply the novel initialization technique
```
python run_sharedpwnet.py 6 2 new_proto_init
```
If you don't specify nothing, a default value for number of prototype and slots is set, and the novel initialization technique is NOT applied.

NOTES:

- At the end of the training the following directories will be created:
    - weights/: where the best models' parameters (among all epochs) are saved for every iteration 
    - results/: where all the models' results are stored in .txt files
    - prototypes/: where all the prototypes found by the models are saved 
    - runs/: to log and visualize training statistics



- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)

Car Racing requirements:
```
pip3 install --upgrade pip
pip install toml
pip install numpy
pip3 install torch torchvision torchaudio
pip install gym'[box2d]'
pip install tqdm
pip install gym==0.24.0
pip install gym-notices==0.0.7
pip install scikit-learn

pip install tensorboard
```
Atari Pong requirements:
```
pip3 install --upgrade pip
pip install gym==0.21.0  
pip install opencv-python
pip3 install torch torchvision torchaudio
pip install atari-py==0.2.9
pip install gym'[atari]'
pip install gym'[accept-rom-license]'
pip install scikit-learn
pip install toml

pip install tensorboard
```
