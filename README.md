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
```
In order to train the networks (all networks are trained for 100 epochs), inside the CarRacing directory of the repo, run:

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet*.py
```

- For modified PW-Net** in which the projection of prototypes is done during training:
```
python run_pwnet**.py
```

- For my prototype net:
```
python run_myprotonet.py
```
-- Note that when running my prototype net, the results obtained by trying NUM_PROTOTYPES = 4,6,9 and NUM_SLOTS_PER_CLASS = 1,2,3 can be found in the myprotonet_results.txt file in the CarRacing directory. The same for the networks pwnet* and pwnet** in which the results are stored in pwnet*_results.txt and pwnet**_results.txt respectively.

-- In order to see the behaviour of the running loss through epochs at each iteration, at the end of the execution of run_myprotonet.py, run_pwnet*.py and run_pwnet**.py, simply run (always inside the CarRacing directory):
```
tensorboard --logdir=runs
```

[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)
