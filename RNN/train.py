import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jax
import optax
from flax.training import train_state
from tqdm import tqdm

from Trainer.trainer import Trainer
from model import MDNRNN, mdn_loss_fn

num_epochs = 20
batch_size = 100
lr = 1e-3


def _loss_fn(params, state, z_t, a_t, z_next):
    log_pi, mu, sigma, _ = state.apply_fn({'params': params}, z_t, a_t)
    return mdn_loss_fn(log_pi=log_pi, mu=mu, sigma=sigma, z_next=z_next)


@jax.jit
def train_step(key, z_t, a_t, z_next, state):
    loss, grads = jax.value_and_grad(_loss_fn)(state.params, state, z_t, a_t, z_next)
    state = state.apply_gradients(grads=grads)
    return state, loss


class MemoryTrainer(Trainer):
    def __init__(self, seed: int = 42):
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'series.npz')
        )
        with np.load(data_path) as data:
            # (N, T, latent_dim) and (N, T, action_dim)
            self.mu = data['mu']
            self.logvar = data['logvar']
            self.action = data['action']

        super().__init__(seed=seed, ckpt_prefix='rnn')

    # Abstract method implementations
    
    def init_model(self):
        z_sample = self.mu[:1]
        a_sample = self.action[:1]
        self.model = MDNRNN()
        params = self.model.init(self.init_key, z_sample, a_sample)
        tx = optax.adam(learning_rate=lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params['params'], tx=tx,
        )

    def _train_epoch(self, key, epoch: int, num_epochs: int):
        indices = np.random.permutation(len(self.mu))
        mu_s = self.mu[indices]
        logvar_s = self.logvar[indices]
        action_s = self.action[indices]

        loss = 0.0
        pbar = tqdm(
            range(0, len(mu_s), batch_size),
            desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=1.0,
        )
        for i in pbar:
            mu_b = mu_s[i:i + batch_size]
            logvar_b = logvar_s[i:i + batch_size]
            action_b = action_s[i:i + batch_size]

            # Reparameterisation trick
            std_b = np.exp(logvar_b / 2)
            z_b = mu_b + np.random.normal(0, 1, size=std_b.shape) * std_b
            z_t = z_b[:, :-1, :]
            z_next = z_b[:, 1:, :]
            a_t = action_b[:, :-1, :]

            key, subkey = jax.random.split(key)
            self.state, loss = train_step(subkey, z_t, a_t, z_next, self.state)
            pbar.set_postfix(loss=f'{loss:.4f}')

        return key, loss


if __name__ == "__main__":
    trainer = MemoryTrainer()
    key = jax.random.key(0)
    if not trainer.checkpoint_exists():
        trainer.train(key, num_epochs=num_epochs)
    else:
        print("Checkpoint found, skipping training")
    