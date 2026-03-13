# JAX-WorldModel


Reimplementation of https://arxiv.org/abs/1803.10122 in JAX.

## Upgrade
Original paper uses gym, but to be compatible with modern development, this repository uses gymnasium.


## How to use
run the followings from repo root
1. Collect image rollouts
```bash
python -m VAE.collect_data
```

2. Create the latent/action series dataset
```bash
python -m RNN.create_series_data
```
