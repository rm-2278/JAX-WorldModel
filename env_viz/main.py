import gymnasium as gym
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np


env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)

obs, info = env.reset()

data_storage = []

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    resized_obs = jax.image.resize(obs, (64, 64, 3), method="bilinear")
    data_storage.append(resized_obs / 255.) # Normalize
    if terminated or truncated:
        obs, info = env.reset()
        
dataset = np.array(data_storage)
print(dataset.shape) # 1000, 64, 64, 3
    
    