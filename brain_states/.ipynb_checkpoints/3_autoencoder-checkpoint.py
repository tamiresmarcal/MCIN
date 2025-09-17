# Env
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np
import gc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import pandas as pd


# Read data
df = pd.read_parquet("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/all_dfcs_flattened.parquet")
X_tensor = torch.tensor(df.values)


# Define the Autoencoder
input_dim = X_tensor.shape[1]

for latent_dim in [2, 3, 5, 8, 13, 21, 34, 55]:
#latent_dim = 5
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim=input_dim, latent_dim=latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Linear(50, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 50),
                nn.ReLU(),
                nn.Linear(50, input_dim),
                nn.Sigmoid()
            )
    
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    
    # Instantiate and train
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Traning 
    n_epochs = 150
    losses = []
    for epoch in range(n_epochs):
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        losses.append(loss.item())
    
    # Save Autoencoder Training Loss plot
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Autoencoder Training Loss")
    plt.savefig(f"/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/Autoencoder_Training_Loss_{latent_dim}.png", dpi=100, bbox_inches='tight')
    
    # Get and save compressed representations
    with torch.no_grad():
        latent = model.encoder(X_tensor).numpy()
    df = pd.DataFrame(latent)
    df.to_csv(f"/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/latent_representations_{latent_dim}.csv")