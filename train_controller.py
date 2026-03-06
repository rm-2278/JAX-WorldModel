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
    
vae = vision_trainer.state
rnn = ??
controller = Controller()
controller_params = controller.init(key, jnp.zeros((1, 32)), jnp.zeros((1, 256)))
controller = train_state.TrainState.create(apply_fn=controller.apply, params=controller_params)

@jax.jit
def get_action(vae_params, rnn_params, controller_params, h, obs):
    z = vision_trainer.model.apply({'params': vae_params}, obs, method=vision_trainer.model.encode)
    a = controller.apply({'params': controller_params}, z, h)
    next_h = rnn.apply_fn({'params': rnn_params}, z, a, h, method=rnn.step)
    return a, next_h

def rollout():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    obs, _ = env.reset()
    obs = process_frame(obs)
    obs = np.expand_dims(obs, axis=0)
    done = False
    
    h = rnn.initialize_carry(jax.random.key(0), (1, 256))
    total_reward = 0
    
    while not done:
        a, next_h = get_action(vae.params, rnn.params, controller.params, h, obs)
        obs, reward, terminated, truncated, _ = env.step(np.array(a[0]))
        done = terminated or truncated
        total_reward += reward
        h = next_h