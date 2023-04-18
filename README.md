# intrepretable_rl_private_caterina


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
```
Inside the CarRacing directory of the repo, simply run:

- For Prototype-Wrapper Network* (PW-Net*) (trainable parameters version of PW-Net) from paper *"Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes"*[^1]:
```
python run_pwnet*.py
```

- For modified PW-Net** in which projection of prototypes is done during training:
```
python run_pwnet**.py
```

- For my prototype net:
```
python run_myprotonet.py
```


[^1]: Kenny, E.M., Tucker, M. and Shah, J., Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes. In *The Eleventh International Conference on Learning Representations.* Kigali, Rwanda, 2023. (Spotlight, notable paper)