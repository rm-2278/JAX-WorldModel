import jax
import jax.numpy as jnp
import flax.linen as nn

class Controller(nn.Module):
    
    @nn.compact
    def __call__(self, z, h):
        x = jnp.concatenate([z, h], axis=-1)
        x = nn.Dense(features=3)(x)
        x = nn.tanh(x)
        x = x.at[..., 1].set((x[..., 1] + 1.) / 2.)
        x = x.at[..., 2].set(jnp.clip(x[..., 2], 0., 1.))
        return x