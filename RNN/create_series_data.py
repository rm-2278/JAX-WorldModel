import sys
sys.path.append('..') 
import os
import glob
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from VAE import *

DIR_NAME = '../data'
DATADIR_NAME = '../data/shards'

@jax.jit
def encode_batch(params, batch):
    batch = batch.astype(jnp.float32) / 255.    # jax converts np to jax array automatically
    mu, logvar = vae_model.apply(params, batch, method=vae_model.encode)
    return mu, logvar

key = random.key(0)
trainer = Trainer(latent_dim=32)
if not trainer.checkpoint_exists():
    trainer.train(key)
else:
    trainer.load_model()

vae_model = trainer.model

files = glob.glob(os.path.join(DATADIR_NAME, '*.npz'))
mu_list = []
logvar_list = []
action_list = []

BATCH_SIZE = 100
EPISODE_LEN = 1000

for file_path in tqdm(files):
    try:
        with np.load(file_path) as data:
            obs = data['obs']
            actions = data['actions']
    except Exception as e:
        print(f"bad file {file_path}: {e}")
        continue
    
    mu_file = []
    logvar_file = []
    num_frames = obs.shape[0]
    
    for i in range(0, num_frames, BATCH_SIZE):
        obs_batch = obs[i:i+BATCH_SIZE]
        mu, logvar = encode_batch(trainer.state.params, obs_batch)
        
        mu_file.append(np.array(mu))
        logvar_file.append(np.array(logvar))
    
    # Assume all data 1000 rollout
    num_episodes_in_file = num_frames // EPISODE_LEN
    
    flat_mu = np.concatenate(mu_file, axis=0)
    flat_logvar = np.concatenate(logvar_file, axis=0)
    
    # Reshapen to make it 1000 frames each batch
    reshaped_mu = flat_mu.reshape(num_episodes_in_file, EPISODE_LEN, 32)
    reshaped_logvar = flat_logvar.reshape(num_episodes_in_file, EPISODE_LEN, 32)
    reshaped_action = actions.reshape(num_episodes_in_file, EPISODE_LEN, -1) 
    
    mu_list.append(reshaped_mu)
    logvar_list.append(reshaped_logvar)
    action_list.append(reshaped_action)

mu_list = np.concatenate(mu_list, axis=0)
logvar_list = np.concatenate(logvar_list, axis=0)
action_list = np.concatenate(action_list, axis=0)

# (10000, 1000, 32)
print(f"Dataset shape: {mu_list.shape}")

np.savez_compressed(
    f"{DIR_NAME}/series.npz",
    mu = mu_list,
    logvar = logvar_list,
    action  = action_list
)

print("Series dataset saved")

    
    
    
