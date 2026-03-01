from flax import linen as nn
import jax
import jax.numpy as jnp



class MDNRNN(nn.Module):
    hidden_dim: int = 256 # 512 for vizdoom
    num_mixtures: int = 5
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, z_t, a_t, carry=None):
        x_t = jnp.concatenate([z_t, a_t], axis=-1)
        
        lstm = nn.LSTMCell(features=self.hidden_dim)
        
        if carry is None:
            batch_size = x_t.shape[0]
            carry = lstm.carry_init(jax.random.key(0), (batch_size, self.hidden_dim))
            
        # nn.scan
        
        # project lstm output to mdn parameters
        mdn_params = nn.Dense(features=325) # (1 + 32 + 32) * 5 = 325
        
        return log_pi, mu, sigma, final_carry