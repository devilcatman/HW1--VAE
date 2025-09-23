#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_grid(images: torch.Tensor, path: str, rows: int, cols: int):
    """
    images: [N, 1, 28, 28] 值域 [0,1]
    """
    images = images.detach().cpu().clamp(0, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        # Encoder：784 -> 400 -> (mu, logvar)
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        # Decoder：z -> 400 -> 784
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total = 0.0
    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)
        if train_mode:
            optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, _, _ = vae_loss(x_hat, x, mu, logvar)
        if train_mode:
            loss.backward()
            optimizer.step()
        total += loss.item()

    n = len(loader.dataset) 
    return total / n

def main():
    parser = argparse.ArgumentParser(description="VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

    device = "cuda" if torch.cuda.is_available() else "cpu"

   
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    for ep in range(1, args.epochs + 1):
        tr_avg = run_epoch(model, train_loader, device, optimizer)
        print(f"Epoch {ep}, Average loss: {tr_avg:.4f}")

    print("訓練完成！")

    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x.view(x.size(0), -1))
    x_hat_img = x_hat.view(-1, 1, 28, 28)

    save_grid(x_hat_img[:16], "reconstruction.png", rows=4, cols=4)
    print("已儲存重建圖檔 reconstruction.png")


if __name__ == "__main__":
    main()

