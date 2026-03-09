from flax import linen as nn
import jax
import jax.numpy as jnp



def mdn_loss_fn(log_pi, mu, sigma, z_next):
    log_N = jax.scipy.stats.norm.logpdf(jnp.expand_dims(z_next, 2), mu, sigma).sum(axis=-1) # (B, T, 5)
    log_P = jax.scipy.special.logsumexp(log_pi + log_N, axis=-1)    # (B, T)
    L = -log_P.mean()
    return L

class MDNRNN(nn.Module):
    hidden_dim: int = 256 # 512 for vizdoom
    num_mixtures: int = 5
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, z_t, a_t, carry=None):
        x_t = jnp.concatenate([z_t, a_t], axis=-1)
        
        # nn.scan makes the LSTMCell sequential
        LSTM = nn.scan(
            nn.LSTMCell,    # The unit cell for LSTM (a function)
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1,  # scan over time dimension (axis 1)
        )
        lstm = LSTM(features=self.hidden_dim)   # features is automatically passed to LSTMCell __init__

        if carry is None:
            batch_size = x_t.shape[0]
            carry = lstm.initialize_carry(jax.random.key(0), (batch_size, x_t.shape[-1]))

        final_carry, outputs = lstm(carry, x_t)  # (B, T, hidden_dim)
        
        # project lstm output to mdn parameters
        mdn_params = nn.Dense(features=325)(outputs) # (1 + 32 + 32) * 5 = 325  (B, T, 325)
        
        log_pi, mu, log_sigma = jnp.split(mdn_params, [1*self.num_mixtures, 33*self.num_mixtures], axis=-1)
        batch_size, time_steps, _ = outputs.shape
        mu = mu.reshape(batch_size, time_steps, self.num_mixtures, -1)  # Last dim latent dim (B, T, 5, 32)
        log_sigma = log_sigma.reshape(batch_size, time_steps, self.num_mixtures, -1)
        
        log_pi = jax.nn.log_softmax(log_pi, axis=-1)
        
        sigma = jnp.exp(log_sigma) + 1e-6   # Ensure positive
        
        return log_pi, mu, sigma, final_carry