import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn

class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        """Encoder Module

        Args:
            x (float): (B, 64, 64, 3) image, 0-1
        """
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=2, padding='VALID')(x) #(31,31,32)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=2, padding='VALID')(x) #(14,14,64)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=2, padding='VALID')(x) #(6,6,128)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(4, 4), strides=2, padding='VALID')(x) #(2,2,256)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.latent_dim*2)(x)
        mu, logvar = jnp.split(x, 2, -1)
        return mu, logvar
        
        
        
class Decoder(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, z):
        """Decoder Module

        Args:
            x (_type_): (B, latent_dim)
        """
        x = nn.Dense(1024)(z)
        x = x.reshape(x.shape[0], 1, 1, -1) #(1,1,1024)
        x = nn.ConvTranspose(features=128, kernel_size=(5,5), strides=2, padding='VALID')(x) #(5,5,128)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(5,5), strides=2, padding='VALID')(x) #(13,13,64)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(6,6), strides=2, padding='VALID')(x) #(30,30,32)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(6,6), strides=2, padding='VALID')(x) #(64,64.3)
        x = nn.sigmoid(x)
        return x
        
        
class VAE(nn.Module):
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        mu, logvar = Encoder(latent_dim=self.latent_dim)(x)
        std = jnp.exp(0.5 * logvar)
        rng = self.make_rng('sample')
        z = mu + random.normal(rng, mu.shape) * std # Reparameterisation
        x = Decoder(latent_dim=self.latent_dim)(z)
        return x, mu, logvar
    
    @nn.compact
    def encode(self, x):
        mu, logvar = Encoder(latent_dim=self.latent_dim)(x)
        return mu, logvar

# x = random.normal(random.key(42), (100, 64, 64, 3))
# vae = VAE()
# params = vae.init(random.key(0), x)
# x, mu, logvar = vae.apply(params, x)
# print(x.shape, mu.shape, logvar.shape)