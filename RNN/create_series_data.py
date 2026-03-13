import glob
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAE import VisionTrainer

DIR_NAME = str(REPO_ROOT / "data")
DATADIR_NAME = str(REPO_ROOT / "data" / "shards")

def main():
    key = random.key(0)
    trainer = VisionTrainer(latent_dim=32)
    if not trainer.checkpoint_exists():
        trainer.train(key)
    else:
        trainer.load_model()

    vae_model = trainer.model

    @jax.jit
    def encode_batch(params, batch):
        batch = batch.astype(jnp.float32) / 255.0  # jax converts np to jax array automatically
        mu, logvar = vae_model.apply(params, batch, method=vae_model.encode)
        return mu, logvar

    files = glob.glob(os.path.join(DATADIR_NAME, "*.npz"))
    mu_list = []
    logvar_list = []
    action_list = []

    batch_size = 100
    episode_len = 1000

    for file_path in tqdm(files):
        try:
            with np.load(file_path) as data:
                obs = data["obs"]
                actions = data["actions"]
        except Exception as e:
            print(f"bad file {file_path}: {e}")
            continue

        mu_file = []
        logvar_file = []
        num_frames = obs.shape[0]

        for i in range(0, num_frames, batch_size):
            obs_batch = obs[i: i + batch_size]
            mu, logvar = encode_batch(trainer.state.params, obs_batch)
            mu_file.append(np.array(mu))
            logvar_file.append(np.array(logvar))

        # Assume all data 1000 rollout
        num_episodes_in_file = num_frames // episode_len

        flat_mu = np.concatenate(mu_file, axis=0)
        flat_logvar = np.concatenate(logvar_file, axis=0)

        # Reshape to make it 1000 frames each batch
        reshaped_mu = flat_mu.reshape(num_episodes_in_file, episode_len, 32)
        reshaped_logvar = flat_logvar.reshape(num_episodes_in_file, episode_len, 32)
        reshaped_action = actions.reshape(num_episodes_in_file, episode_len, -1)

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
        mu=mu_list,
        logvar=logvar_list,
        action=action_list,
    )

    print("Series dataset saved")


if __name__ == "__main__":
    main()

    
    
    
