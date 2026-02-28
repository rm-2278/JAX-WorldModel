import numpy as np
from tqdm import tqdm

num_epochs = 20
batch_size = 100

with np.load('../data/series.npz') as data:
    # (10000, 1000, 32 or 3)
    mu = data['mu']
    logvar = data['logvar']
    action = data['action']
    
for epoch in tqdm(range(num_epochs)):
    indices = np.random.permutation(len(mu))
    mu_shuffled = mu[indices]
    logvar_shuffled = logvar[indices]
    action_shuffled = action[indices]
    
    for i in range(0, len(mu_shuffled), batch_size):
        mu_batch = mu_shuffled[i:i+batch_size]
        logvar_batch = logvar_shuffled[i:i+batch_size]
        action_batch = action_shuffled[i:i+batch_size]
        
        std_batch = np.exp(logvar_batch / 2)
        z_batch = mu_batch + np.random.normal(0, 1, size=std_batch.shape) * std_batch
        z_t = z_batch[:, :-1, :]
        z_next = z_batch[:, 1:, :]
        a_t = action_batch[:, :-1, :]
        
        # Train the model using z_t, a_t to predict z_next (using h_t)
        
    