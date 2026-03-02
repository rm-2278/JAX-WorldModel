import os
import numpy as np
import cv2
import jax
import gymnasium as gym
from model import VAE
from tqdm import tqdm
from joblib import Parallel, delayed

"""Collects the data and stores as npz"""

MAX_FRAMES = 1000   # Embedded in default
ROLLOUTS = 10000  # Change to 200 for debugging
ROLLOUTS_PER_SHARD = 100
TOTAL_SHARDS = ROLLOUTS // ROLLOUTS_PER_SHARD
DIR_NAME = '../data/shards'
NUM_WORKERS = -1    # All workers

os.makedirs(DIR_NAME, exist_ok=True)

def process_frame(obs):
    obs = obs[:84, :, :]    # Crop out the bottom dashboard
    obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)   # preserves channel dimension
    # obs = jax.image.resize(obs, (64, 64, 3), method="bilinear")    # jax in gym loop inefficient
    return obs

def collect_shard(shard_id):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
    obs_list = []
    action_list = []
    done_list = []
    
    for i in range(ROLLOUTS_PER_SHARD):
        obs, _ = env.reset(seed=shard_id*1000+i)
        obs = process_frame(obs)
        
        for _ in range(MAX_FRAMES):
            obs_list.append(obs)
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = process_frame(obs)

            action_list.append(action)
            done_list.append(terminated or truncated)
            if terminated or truncated:
                break
    
    filename = f"{DIR_NAME}/shard_{shard_id}.npz"
    np.savez_compressed(filename, obs=np.stack(obs_list, dtype=np.uint8), 
                        actions=np.stack(action_list, dtype=np.float32), 
                        done=np.stack(done_list, dtype=bool))

if __name__ == "__main__":
    print(f"Starting collection of {TOTAL_SHARDS} shards")
    
    results = Parallel(n_jobs=NUM_WORKERS, verbose=10)(
        delayed(collect_shard)(i) for i in range(TOTAL_SHARDS)
    )
    
    print(f"Done collecting shardss")