Feb 19: Created the repository. Visualized and resized the car-racing environment image.
Feb 20: Created Encoder and Decoder for CVAE
Feb 21: Started on storing data for VAE training
Feb 22: Created a dataloader that randomly generates a batch
Feb 24: Trained the VAE.



Step 2: The MDN-RNN (Memory Model)In the World Models paper, the agent needs to predict the future to make informed decisions. Since the environment is stochastic (random), a simple prediction (like Mean Squared Error) will fail because it averages all possible futures into a blurry mess.Instead, we use a Mixture Density Network (MDN) combined with an RNN.Input: Current latent code $z_t$ and action $a_t$.Output: The probability distribution of the next latent code $z_{t+1}$.Distribution: A mixture of Gaussians (5 mixtures is standard for CarRacing).Phase 1: Pre-process the DataYou cannot train the RNN directly on images—it's too slow. You must use your trained VAE to "compress" your entire dataset of rollouts into sequences of latent vectors.Task: Create a script to iterate through your .npz files, run the VAE encoder to get $z$ (specifically $\mu$), and save a new dataset of sequences.Hint:Your new data should look like: (N, Seq_Len, 35) where 35 = 32 (latent) + 3 (action).Important: Don't forget to save the next latent $z_{t+1}$ as the target for training.Phase 2: Building the MDN-RNN in JAXHere is the architectural blueprint for the Memory model using flax.linen.1. The RNN CoreIn JAX/Flax, handling state in RNNs requires explicit management. The jax.lax.scan primitive is your best friend here—it allows you to efficiently unroll the LSTM over the sequence dimension.Implementation Hint:Pythonimport flax.linen as nn

class MDNRNN(nn.Module):
    hidden_size: int = 256
    num_mixtures: int = 5
    latent_dim: int = 32

    @nn.compact
    def __call__(self, inputs, start_state=None):
        # inputs shape: (batch, seq_len, latent_dim + action_dim)
        
        lstm_cell = nn.LSTMCell(features=self.hidden_size)
        
        # Initialize state if not provided
        if start_state is None:
            batch_size = inputs.shape[0]
            start_state = lstm_cell.initialize_carry(jax.random.key(0), (batch_size, self.hidden_size))

        # Use nn.scan to unroll the LSTM over the sequence (axis 1)
        # This is much faster than a python for-loop
        scan_cell = nn.scan(lstm_cell, variable_broadcast="params", split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        final_state, rnn_outputs = scan_cell(start_state, inputs)
        
        # Now project rnn_outputs to MDN parameters
        # We need output for: Pi, Mu, and Sigma for EACH mixture
        output_dim = self.num_mixtures * (2 * self.latent_dim + 1)
        x = nn.Dense(output_dim)(rnn_outputs)
        
        return x, final_state
2. The MDN Output HeadThe raw output from the Dense layer needs to be split and transformed to represent valid probability distributions:Log-Pi ($\log \pi$): The mixing coefficients. Use log_softmax to ensure they sum to 1 in log-space.Mu ($\mu$): The means. No activation needed.Sigma ($\sigma$): The standard deviations. Must be positive! Apply jnp.exp or nn.softplus.3. The Loss Function (Negative Log Likelihood)This is the most mathematically dense part. You want to maximize the likelihood of the true $z_{t+1}$ appearing in your predicted mixture distribution.We minimize the Negative Log Likelihood (NLL).$$Loss = - \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(z_{t+1} | \mu_k, \sigma_k) \right)$$JAX Stability Hint:Computing the sum of probabilities directly often leads to NaN (underflow). Work in log-space as much as possible using jax.scipy.special.logsumexp.A rough sketch of the loss logic:Calculate the log-probability of $z_{t+1}$ for each Gaussian component $k$ separately (Log-Normal).Add the mixing coefficients: log_prob_k + log_pi_k.Combine them using logsumexp across the mixture dimension.Negate the result and take the mean.Recommended LibrariesDistrax: If you prefer not to implement the Gaussian math manually, DeepMind's distrax library (the JAX equivalent of TensorFlow Probability) has distrax.MixtureSameFamily which works beautifully here.Optax: Continue using this for your optimizer (Adam is standard).




Later on...
Instead of random uniform, we must use random policy instead
We can probably further optimize training of VAE.