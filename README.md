# interpretable_rl_private_caterina


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

- First run this inside CarRacing directory to collect the data needed to train the neural networks:
```
python collect_data.py
```

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

- For my prototype net (NUM_PROTOTYPES = 4,6 and NUM_SLOTS_PER_CLASS = 2):
```
python run_myprotonet.py
```

-- NOTES:
At the end of the training the following directories will be created:
- weights/: where the best models' parameters (among all epochs) are saved for every iteration (NUM_ITERATIONS=5)
- results/: where all the models' results are stored in .txt files
- prototypes/: where all the prototypes found by the models are saved 
- runs/: to log and visualize training statistics



-- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet_star.py and run_pwnet_star_star.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)
