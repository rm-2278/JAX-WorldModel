from VAE import VAE, VisionTrainer
from VAE.collect_data import process_frame
from RNN import MDNRNN, MemoryTrainer
from Controller import Controller

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints, train_state

import gymnasium as gym

key = jax.random.key(0)
vision_trainer = VisionTrainer(latent_dim=32)
if not vision_trainer.checkpoint_exists():
    vision_trainer.train(key)
else:
    vision_trainer.load_model()

memory_trainer = MemoryTrainer()
if not memory_trainer.checkpoint_exists():
    memory_trainer.train(key)
else:
    memory_trainer.load_model()


vae = vision_trainer.state
rnn = memory_trainer.state
controller = Controller()
controller_params = controller.init(key, jnp.zeros((1, 32)), jnp.zeros((1, 256)))

@jax.jit
def get_action(vae_params, rnn_params, controller_params, h, obs):
    (mu, logvar) = vision_trainer.model.apply({'params': vae_params}, obs, method=vision_trainer.model.encode)
    z = mu + jnp.exp(logvar / 2.0)  # Could make it deterministic
    a = controller.apply({'params': controller_params}, z, h)
    _, _, _, next_h = memory_trainer.model.apply({'params': rnn_params}, z, a, h, method=memory_trainer.model.step)
    return a, next_h

def rollout(controller_params):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    obs, _ = env.reset()
    obs = process_frame(obs)
    obs = np.expand_dims(obs, axis=0)
    done = False
    
    h = memory_trainer.model.initialize_carry(jax.random.key(0), (1, 256))
    total_reward = 0
    
    while not done:
        a, next_h = get_action(vae.params, rnn.params, controller_params, h, obs)
        obs, reward, terminated, truncated, _ = env.step(np.array(a[0]))
        obs = process_frame(obs)
        obs = np.expand_dims(obs, axis=0)
        done = terminated or truncated
        total_reward += reward
        h = next_h
        
    return total_reward