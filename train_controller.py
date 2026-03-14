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
controller_params = controller.init(key, jnp.zeros((1, 32)), jnp.zeros((1, 256)))["params"]

@jax.jit
def get_action(rng, vae_params, rnn_params, controller_params, carry, obs):
    obs = obs.astype(jnp.float32) / 255.0
    mu, logvar = vision_trainer.model.apply({"params": vae_params}, obs, method=vision_trainer.model.encode)

    rng, eps_key = jax.random.split(rng)
    eps = jax.random.normal(eps_key, shape=mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)

    h = carry[1]  # carry = (c, h)
    a = controller.apply({"params": controller_params}, z, h)

    _, _, _, next_carry = memory_trainer.model.apply(
        {"params": rnn_params},
        z,
        a,
        carry,
        method=memory_trainer.model.step,
    )
    return rng, a, next_carry

def rollout(controller_params):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    obs, _ = env.reset()
    obs = process_frame(obs)
    obs = np.expand_dims(obs, axis=0)
    done = False

    action_dim = 3
    in_dim = 32 + action_dim
    carry = memory_trainer.model.apply(
        {"params": rnn.params},
        jax.random.key(0),
        (1, in_dim),
        method=memory_trainer.model.initialize_carry,
    )
    total_reward = 0.0

    rng = jax.random.key(0)
    
    while not done:
        rng, a, carry = get_action(rng, vae.params, rnn.params, controller_params, carry, obs)
        obs, reward, terminated, truncated, _ = env.step(np.array(a[0]))
        obs = process_frame(obs)
        obs = np.expand_dims(obs, axis=0)
        done = terminated or truncated
        total_reward += reward
        
    return total_reward