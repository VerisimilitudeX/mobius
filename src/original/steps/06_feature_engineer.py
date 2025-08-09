#!/usr/bin/env python3
"""
06_feature_engineer.py

Purpose:
  - Reads a cleaned CSV (row=sample, columns=features, plus Condition).
  - Trains a Variational Autoencoder (VAE) for dimensionality reduction instead of PCA.
  - During training, records the reconstruction loss and KL divergence.
  - After training, prints the dimensions of the latent features produced.
  - It saves:
      • The VAE loss convergence curves as "vae_loss_curve.png"
      • The transformed data CSV ("transformed_data.csv")
      • A “head” file ("transformed_data_head.csv") showing the first few rows of latent features.
      
Usage example:
  python 06_feature_engineer.py \
    --csv /Volumes/T9/EpiMECoV/processed_data/cleaned_data.csv \
    --out /Volumes/T9/EpiMECoV/processed_data/transformed_data.csv \
    --latent_dim 64 \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.1 \
    --use_scale True
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
import matplotlib.pyplot as plt

class BetaDataset(Dataset):
    """Simple PyTorch Dataset that extracts numeric columns as X, ignoring 'Condition'."""
    def __init__(self, df, condition_col="Condition"):
        self.condition_col = condition_col
        self.features = df.drop(columns=[condition_col]).values.astype(np.float32)
        self.labels = df[condition_col].values  # not used for VAE target
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

class VAE(nn.Module):
    """
    Variational Autoencoder with one hidden layer.
    """
    def __init__(self, input_dim, latent_dim=64, dropout_p=0.0):
        super(VAE, self).__init__()
        hidden_dim = 256
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_p)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout(h1)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h2 = torch.relu(self.fc2(z))
        h2 = self.dropout(h2)
        return self.fc3(h2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: MSE
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to cleaned CSV (row=sample, columns=features+Condition).")
    parser.add_argument("--out", required=True,
                        help="Output CSV path for transformed data (latent features + Condition).")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_scale", type=bool, default=False,
                        help="Whether to standard-scale the numeric columns.")
    parser.add_argument("--condition_col", default="Condition")
    args = parser.parse_args()

    print("=== Feature Engineering (VAE) ===")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Could not find CSV => {args.csv}")

    # 1) Load DataFrame
    df = pd.read_csv(args.csv)
    print(f"Input shape = {df.shape}")

    if args.condition_col not in df.columns:
        raise ValueError(f"No '{args.condition_col}' column found in DataFrame.")

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2) Separate numeric vs. Condition
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if args.condition_col in numeric_cols:
        numeric_cols.remove(args.condition_col)
    keep_cols = numeric_cols + [args.condition_col]
    df = df[keep_cols]
    print(f"Using numeric columns + {args.condition_col}, shape = {df.shape}")

    # 3) Optional scaling
    if args.use_scale:
        if not HAVE_SKLEARN:
            raise ImportError("scikit-learn is not installed => cannot use --use_scale.")
        sc = StandardScaler()
        df_num = sc.fit_transform(df[numeric_cols].values)
        df[numeric_cols] = df_num
        print("[INFO] Applied standard scaling to numeric columns.")

    # 4) Create Dataset and determine input dimension
    dataset = BetaDataset(df, condition_col=args.condition_col)
    input_dim = dataset.features.shape[1]
    print(f"[INFO] VAE input dimension = {input_dim}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5) Build VAE, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim, latent_dim=args.latent_dim, dropout_p=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Record training losses for convergence curves
    recon_losses = []
    kl_losses = []
    total_losses = []

    print(f"Training VAE for {args.epochs} epochs on {len(dataset)} samples (batch size = {args.batch_size}).")
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss_ep = 0.0
        total_recon_ep = 0.0
        total_kl_ep = 0.0
        for X_batch in dataloader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(X_batch)
            recon_loss, kl_loss = loss_function(recon, X_batch, mu, logvar)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss_ep += loss.item() * X_batch.size(0)
            total_recon_ep += recon_loss.item() * X_batch.size(0)
            total_kl_ep += kl_loss.item() * X_batch.size(0)
        avg_loss = total_loss_ep / len(dataset)
        avg_recon = total_recon_ep / len(dataset)
        avg_kl = total_kl_ep / len(dataset)
        total_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        print(f"[Epoch {ep}/{args.epochs}] Total Loss = {avg_loss:.6f}, Recon Loss = {avg_recon:.6f}, KL Loss = {avg_kl:.6f}")

    # Plot and save the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epochs + 1), total_losses, marker='o', label="Total Loss")
    plt.plot(range(1, args.epochs + 1), recon_losses, marker='s', label="Reconstruction Loss")
    plt.plot(range(1, args.epochs + 1), kl_losses, marker='^', label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Convergence")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(os.path.dirname(args.out), "vae_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"[INFO] Loss convergence curves saved to {loss_curve_path}")

    # 6) Generate latent features for all samples (using the mean, mu)
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(dataset.features).to(device)
        _, mu_all, _ = model(X_all)
        latents = mu_all.cpu().numpy()
    print(f"[INFO] Latent features shape: {latents.shape}")

    latent_df = pd.DataFrame(latents, columns=[f"VAE_{i}" for i in range(args.latent_dim)])
    latent_df[args.condition_col] = df[args.condition_col].values
    head_df = latent_df.head(10)
    head_out = os.path.join(os.path.dirname(args.out), "transformed_data_head.csv")
    head_df.to_csv(head_out, index=False)
    print(f"[INFO] Head of latent features saved to {head_out}")

    latent_df.to_csv(args.out, index=False)
    print(f"[INFO] Transformed data saved to {args.out}")

    print("=== Done feature engineering with VAE. ===")


if __name__ == "__main__":
    main()