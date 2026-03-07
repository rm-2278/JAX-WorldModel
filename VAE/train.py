import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax.training import train_state, checkpoints
import optax
import glob
from tqdm import tqdm
import multiprocessing as mp
import subprocess

from model import VAE
from Trainer.trainer import Trainer


epochs = 100
batch_size = 100
lr = 1e-4
latent_dim = 32

DIR_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'shards'))



class Dataloader:
    def __init__(self, batch_size=100):
        self.files = glob.glob(os.path.join(DIR_NAME, '*.npz'))
        self.batch_size = batch_size
        self.total_files = len(self.files)
        
        self.total_steps = (self.total_files * 100 * 1000) // batch_size  # Approximate
        
    def __iter__(self):
        np.random.shuffle(self.files)   # Shuffle every epoch
        buffer = np.empty((0, 64, 64, 3), dtype=np.uint8)
        
        for file_path in self.files:
            try:
                with np.load(file_path) as data:
                    obs = data['obs']
            except Exception as e:
                print(f"bad file {file_path}: {e}")
                continue
            
            np.random.shuffle(obs)  # Shuffle within file
            
            if buffer.shape[0] == 0:
                buffer = obs
            else:
                buffer = np.concatenate([buffer, obs], axis=0)
                
            while len(buffer) >= self.batch_size:
                batch = buffer[:self.batch_size]
                buffer = buffer[self.batch_size:]
                
                yield batch
    
    def __len__(self):
        return self.total_steps

class BackgroundGenerator:
    def __init__(self, generator, max_prefetch=10):
        self.generator = generator
        self.queue = mp.Queue(max_prefetch)
        self.process = mp.Process(target=self._fill_queue)  # Using 1 CPU core as background
        self.process.start()
        
    def _fill_queue(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)
    
    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            self.process.join()
            raise StopIteration
        return next_item
    
    def __iter__(self):
        return self

def loss_fn(params, state, key, imgs, kl_tolerance=0.5):
    recon, mu, logvar = state.apply_fn(params, imgs, rngs={'sample': key})
    recon_loss = optax.l2_loss(recon, imgs).sum(axis=(1, 2, 3)).mean() # L2 loss
    kl = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    kl_loss = jnp.maximum(kl, kl_tolerance*latent_dim).mean()     # KL free bits, prevent collapse
    return recon_loss + kl_loss, (recon_loss, kl_loss)
    
@jax.jit
def train_step(key, batch, state):
    batch = batch.astype(jnp.float32) / 255. # Normalization within GPU
    (loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state, key, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


class VisionTrainer(Trainer):
    def __init__(self, latent_dim: int = 32, seed: int = 42):
        # Set VAE-specific attrs BEFORE super().__init__, because
        # super().__init__ calls self.init_model() which needs them.
        self.latent_dim = latent_dim
        self.img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'media'))
        os.makedirs(self.img_dir, exist_ok=True)

        self.dataloader = Dataloader()
        try:
            self.example_img = next(iter(self.dataloader))
        except StopIteration:
            raise RuntimeError('No data found in dataloader')

        super().__init__(seed=seed, ckpt_prefix='vae')

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def init_model(self):
        self.model = VAE(latent_dim=self.latent_dim)
        params = self.model.init(self.init_key, self.example_img)
        optimizer = optax.adam(learning_rate=lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer,
        )

    def _train_epoch(self, key, epoch: int, num_epochs: int):
        dataloader = BackgroundGenerator(Dataloader(batch_size=batch_size))
        loss = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=1.0) as pbar:
            for batch in pbar:
                key, subkey = jax.random.split(key)
                self.state, loss = train_step(subkey, batch, self.state)
                pbar.set_postfix(loss=f'{loss:.4f}')
        return key, loss

    # ------------------------------------------------------------------
    # VAE-specific epoch hook: save comparison images, then checkpoint
    # ------------------------------------------------------------------

    def _on_epoch_end(self, epoch: int):
        self.save_comparison(epoch)
        super()._on_epoch_end(epoch)   # saves checkpoint

    def save_comparison(self, step: int):
        test_imgs = self.example_img[:8].astype(np.float32) / 255.
        recon, _, _ = self.state.apply_fn(self.state.params, test_imgs, rngs={'sample': random.key(0)})
        recon = np.array(recon)
        original = np.array(test_imgs)

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0][i].imshow(original[i])
            axes[0][i].axis('off')
            axes[0][i].set_title('Orig')
            axes[1][i].imshow(recon[i])
            axes[1][i].axis('off')
            axes[1][i].set_title('Recon')

        plt.tight_layout()
        plt.savefig(f'{self.img_dir}/debug_epoch_{step+1}.png')
        plt.close()


            
if __name__ == "__main__":
    # Collect data if no shards exist
    shard_files = glob.glob(os.path.join(DIR_NAME, '*.npz'))
    if len(shard_files) == 0:
        print("No data shards found. Collecting data...")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "collect_data.py")], check=True)
        
    trainer = Trainer(latent_dim=latent_dim)    # 32 for car racing, 64 for doom
    key = random.key(0)
    if not trainer.checkpoint_exists():
        trainer.train(key)
    else:
        print("Checkpoint found, skipping training")
