# interpretable_rl_private_caterina

### Car Racing
Requirements:
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

- First run this inside CarRacing directory to collect the data needed to train the neural networks:
```
python collect_data.py
```


Then, in order to train the networks (all networks are trained for 100 epochs), always inside the CarRacing directory of the repo, run:

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet_star.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet_star_star.py
```

- For my prototype net you have to specify the number of prototypes that the network has to learn and the number of slots per class:
```
python run_myprotonet.py n_proto 6 n_slots 2
```


- NOTES:

- At the end of the training the following directories will be created:
    - weights/: where the best models' parameters (among all epochs) are saved for every iteration (NUM_ITERATIONS=5)
    - results/: where all the models' results are stored in .txt files
    - prototypes/: where all the prototypes found by the models are saved 
    - runs/: to log and visualize training statistics



- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```
### Bipedal Walker

- First run this inside BipedalWalker directory to collect the data needed to train the neural networks:
```
python collect_data.py
```

Then, in order to train the networks (all networks are trained for 100 epochs), always inside the BipedalWalker directory of the repo, run:

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet_star.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet_star_star.py
```

- For my prototype net you have to specify the number of prototypes that the network has to learn and the number of slots per class:
```
python run_myprotonet.py n_proto 6 n_slots 2
```


- NOTES:

- At the end of the training the following directories will be created:
    - weights/: where the best models' parameters (among all epochs) are saved for every iteration (NUM_ITERATIONS=5)
    - results/: where all the models' results are stored in .txt files
    - prototypes/: where all the prototypes found by the models are saved 
    - runs/: to log and visualize training statistics



- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

### Atari Pong
Requirements:
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

- First run this inside AtariPong directory to collect the data needed to train the neural networks:
```
python collect_data.py
```


Then, in order to train the networks (all networks are trained for 100 epochs), always inside the AtariPong directory of the repo, run:
- PW-Net from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet.py
```

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet_star.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet_star_star.py
```

- For my prototype net you have to specify the number of prototypes that the network has to learn and the number of slots per class:
```
python run_myprotonet.py n_proto 6 n_slots 2
```

- NOTES:

- At the end of the training the following directories will be created:
    - weights/: where the best models' parameters (among all epochs) are saved for every iteration (NUM_ITERATIONS=3)
    - results/: where all the models' results are stored in .txt files
    - prototypes/: where all the prototypes found by the models are saved 
    - runs/: to log and visualize training statistics



- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)

### Lunar Lander
- First run this inside LunarLander directory to collect the data needed to train the neural networks:
```
python collect_data.py
```


Then, in order to train the networks (all networks are trained for 100 epochs), always inside the LunarLander directory of the repo, run:
- PW-Net from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet.py
```

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet_star.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet_star_star.py
```

- For my prototype net you have to specify the number of prototypes that the network has to learn and the number of slots per class:
```
python run_myprotonet.py n_proto 6 n_slots 2
```

- NOTES:

- At the end of the training the following directories will be created:
    - weights/: where the best models' parameters (among all epochs) are saved for every iteration (NUM_ITERATIONS=3)
    - results/: where all the models' results are stored in .txt files
    - prototypes/: where all the prototypes found by the models are saved 
    - runs/: to log and visualize training statistics



- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)