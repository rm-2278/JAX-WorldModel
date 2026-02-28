import gymnasium as gym
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np


env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)

obs, info = env.reset()

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(obs)
ax.set_title("CarRacing 64x64")
plt.axis('off')

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(np.array([-0.02, 0.4, 0.]))
    resized_obs = jax.image.resize(obs, (64, 64, 3), method="bilinear")
    display_obs = jnp.clip(resized_obs, 0, 255).astype(jnp.uint8)
    img.set_data(display_obs)
    plt.draw()
    plt.pause(0.001)
    if terminated or truncated:
        obs, info = env.reset()
        
plt.ioff()
plt.close()
env.close()
    
    