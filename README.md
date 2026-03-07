# JAX-WorldModel


Reimplementation of https://arxiv.org/abs/1803.10122 in JAX.

## Upgrade
Original paper uses gym, but to be compatible with modern development, this repository uses gymnasium.


## How to use
1. Go into VAE folder, and execute
'''bash
python collect_data.py
'''

2. Go into RNN folder, change VAE train.py file to add . and execute
'''bash
python create_series_data.py
'''
