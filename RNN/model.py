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
            carry = lstm.initialize_carry(jax.random.key(0), (batch_size, 35))  # Pass in the input for first time step
            
        # nn.scan
        def scan_fn(carry, x):
            new_carry, out = lstm(carry, x)
            return new_carry, out
        
        final_carry, outputs = nn.scan( # (B, T, hidden_dim)
            scan_fn,   
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1   # Time dimension, take at 1st dimension and stack as 1st dimension
            )(carry, x_t)
        
        # project lstm output to mdn parameters
        mdn_params = nn.Dense(features=325)(outputs) # (1 + 32 + 32) * 5 = 325  (B, T, 325)
        
        log_pi, mu, log_sigma = jnp.split(mdn_params, [1*self.num_mixtures, 33*self.num_mixtures], axis=-1)
        batch_size, time_steps, _ = outputs.shape
        mu = mu.reshape(batch_size, time_steps, self.num_mixtures, -1)  # Last dim latent dim
        log_sigma = log_sigma.reshape(batch_size, time_steps, self.num_mixtures, -1)
        
        log_pi = jax.nn.log_softmax(log_pi, axis=-1)
        
        sigma = jnp.exp(log_sigma) + 1e-6   # Ensure positive
        
        return log_pi, mu, sigma, final_carry