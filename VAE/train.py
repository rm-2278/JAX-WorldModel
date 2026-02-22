import gymnasium as gym
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax

from model import VAE

epochs = 100
batch_size = 100

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)

obs, info = env.reset()

# Collecting data
data_storage = []
rollout = 0
while rollout < 10000:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    resized_obs = jax.image.resize(obs, (64, 64, 3), method="bilinear")
    data_storage.append(resized_obs / 255.) # Normalize
    if terminated or truncated:
        rollout += 1
        obs, info = env.reset()
        
dataset = np.array(data_storage)
        
def train(model, params):
    for epoch in range(epochs):
        for i in range(batch_size):
            action = 


def loss_fn(key, model, params, batch):
    imgs, _ = batch
    recon_logits, mu, logvar = model.apply({'params': params}, key, imgs)
    recon_loss = optax.sigmoid_binary_cross_entropy(recon_logits, imgs).sum(axis=(1, 2)).mean() # Numerically more stable, applies sigmoid
    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1).mean()
    return recon_loss + kl_loss

example_image = data_storage[0]

vae = VAE(latent_dim=32)
params = vae.init(random.key(0), example_image)
train(vae, params)