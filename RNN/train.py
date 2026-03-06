import numpy as np
import jax
from tqdm import tqdm
import optax
from flax.training import train_state, checkpoints
import os

from model import MDNRNN, mdn_loss_fn

num_epochs = 20
batch_size = 100

ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoint'))
os.makedirs(ckpt_dir, exist_ok=True)

with np.load('../data/series.npz') as data:
    # (10000, 1000, 32 or 3)
    mu = data['mu']
    logvar = data['logvar']
    action = data['action']

z_sample = mu[:1]
a_sample = action[:1]
model = MDNRNN()
params = model.init(jax.random.key(0), z_sample, a_sample)
tx = optax.adam(learning_rate=1e-3)
state = train_state.TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)

def loss_fn(params, state, z_t, a_t, z_next):
    log_pi, mu, sigma, _ = state.apply_fn({'params': params}, z_t, a_t)
    loss = mdn_loss_fn(log_pi=log_pi, mu=mu, sigma=sigma, z_next=z_next)
    return loss

@jax.jit
def train_step(key, z_t, a_t, z_next, state):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state, z_t, a_t, z_next)
    state = state.apply_gradients(grads=grads)
    return state, loss

key = jax.random.key(0)
for epoch in tqdm(range(num_epochs)):
    indices = np.random.permutation(len(mu))
    mu_shuffled = mu[indices]
    logvar_shuffled = logvar[indices]
    action_shuffled = action[indices]
    
    for i in range(0, len(mu_shuffled), batch_size):
        mu_batch = mu_shuffled[i:i+batch_size]
        logvar_batch = logvar_shuffled[i:i+batch_size]
        action_batch = action_shuffled[i:i+batch_size]
        
        std_batch = np.exp(logvar_batch / 2)
        z_batch = mu_batch + np.random.normal(0, 1, size=std_batch.shape) * std_batch
        z_t = z_batch[:, :-1, :]
        z_next = z_batch[:, 1:, :]
        a_t = action_batch[:, :-1, :]
        
        # Train the model using z_t, a_t to predict z_next (using h_t)
        key, subkey = jax.random.split(key)
        state, loss = train_step(subkey, z_t, a_t, z_next, state)
    
    # Save checkpoint after each epoch
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, step=epoch, target=state.params, prefix='rnn', overwrite=True)

print("Training complete. Model saved to checkpoint/")
    