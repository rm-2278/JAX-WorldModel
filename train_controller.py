from VAE import VAE, Trainer
from VAE.collect_data import process_frame
from RNN import MDNRNN
from Controller import Controller

import jax
import jax.numpy as np
from flax.training import checkpoints, train_state

import gymnasium as gym

key = jax.random.key(0)
trainer = Trainer(latent_dim=32)
if not trainer.checkpoint_exists():
    trainer.train(key)
else:
    trainer.load_model()
    
vae = trainer.state
rnn = ??
controller = Controller()

@jax.jit
def get_action(vae_params, rnn_params, controller_params, h, obs):
    z = vae.apply({'params': vae_params}, obs, method=vae.encode)
    a = controller.apply({'params': controller_params}, z, h)
    next_h = rnn.apply_fn({'params': rnn_params}, z, a, h, method=rnn.step)
    return a, next_h

def rollout():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    obs, _ = env.reset()
    obs = process_frame(obs)
    done = False
    
    h = rnn.initialize_carry(jax.random.key(0), (1, 35))
    total_reward = 0
    
    while not done:
        a, next_h = get_action(vae.params, rnn.params, controller.params, h, obs)
        obs, reward, terminated, truncated, _ = env.step(np.array(a))
        done = terminated or truncated
        total_reward += reward