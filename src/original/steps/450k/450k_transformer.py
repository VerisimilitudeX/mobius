#!/usr/bin/env python3
"""
Epigenomic Transformer Pipeline with MoE + ACT + Masked Pretraining
====================================================================

Usage (quick local test):
  python epigenomic_transformer.py --csv /Users/username/Downloads/output.txt \
                                   --quick_test 1

Usage (full HPC run):
  python epigenomic_transformer.py --csv /path/to/filtered_biomarker_matrix.csv \
                                   --quick_test 0

Key Steps:
 - Loads CSV with shape [Samples x Features], last column = 'Condition'
 - (Optional) transpose if your data is [Features x Samples], see the commented line
 - Masked Pretraining (randomly mask ~15% of CpG values)
 - Fine-tune on classification with mixture-of-experts feed-forward + ACT gating
 - Logs more often so you can see partial progress
 - If 'quick_test=1', it uses fewer epochs, fewer hidden dims, smaller chunk, etc.
 - Else, it uses recommended settings for large-scale HPC training.

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import random

###############################################################################
# DataSet + Collate
###############################################################################
class MethylationCSVDataset(Dataset):
    """
    A dataset that loads the entire CSV into memory. Expects:
        - 'Condition' as the last column
        - The rest columns as numeric features
    If the user sets quick_test=1, we further subsample rows & features to run quickly.
    """
    def __init__(self, csv_path, quick_test=False, logger=print):
        self.logger = logger
        self.logger(f"[INFO] Loading CSV => {csv_path}")
        df = pd.read_csv(csv_path)
        self.logger(f"[INFO] Raw shape from CSV: {df.shape}")

        if "Condition" not in df.columns:
            raise ValueError("No 'Condition' column found in the CSV. Please ensure last col is Condition.")

        # If your CSV has shape [Features x Samples], you may need to transpose.
        # We avoid automatic transpose here to prevent accidental mishandling.

        # Condition as the last column
        cond_values = df["Condition"].values.astype(str)
        self.classes = sorted(np.unique(cond_values))
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.y = np.array([self.class2idx[c] for c in cond_values], dtype=np.int64)

        df_feat = df.drop(columns=["Condition"])
        # Convert to float
        Xmat = df_feat.values.astype(np.float32)
        # Per paper: z-score normalize each sample's beta-values
        eps = 1e-8
        row_mean = Xmat.mean(axis=1, keepdims=True)
        row_std = Xmat.std(axis=1, keepdims=True) + eps
        Xmat = (Xmat - row_mean) / row_std
        self.logger(f"[INFO] Feature matrix shape before quick_test: {Xmat.shape}")

        if quick_test:
            # Subsample rows & cols drastically for a quick run
            # e.g. keep first 200 samples, first 500 features
            n_rows = min(200, Xmat.shape[0])
            n_cols = min(500, Xmat.shape[1])
            Xmat = Xmat[:n_rows, :n_cols]
            self.y = self.y[:n_rows]
            self.logger("[INFO] Quick test => subsample to shape: "
                        f"{n_rows} x {n_cols}")

        # Convert to torch float
        self.X = torch.from_numpy(Xmat).float()
        self.logger(f"[INFO] Final dataset shape => {self.X.shape}, Y => {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################################################
# Autoencoder for Denoising/Feature Extraction (keeps 1280 dims via recon)
###############################################################################
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=1280, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.GELU(),
            nn.Linear(1024, latent_dim), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.GELU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

def run_autoencoder_pretrain(tensor_X, batch_size=64, epochs=50, lr=1e-3, device=None, logger=print):
    """
    Train AE on tensor_X (shape [N, F]). Return reconstructed tensor with same shape.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(input_dim=tensor_X.shape[1], latent_dim=128).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(tensor_X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            # lightweight denoising
            noise = 0.01 * torch.randn_like(xb)
            noisy = xb + noise
            opt.zero_grad()
            recon = model(noisy)
            loss = mse(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        logger(f"[AE] epoch {ep+1}/{epochs} loss={ep_loss/len(dl):.6f}")
    # reconstruct full tensor
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, tensor_X.size(0), batch_size):
            xb = tensor_X[i:i+batch_size].to(device)
            out.append(model(xb).cpu())
        recon_all = torch.cat(out, dim=0)
    torch.save(model.state_dict(), "autoencoder.pth")
    logger("[AE] Saved autoencoder.pth and produced reconstructed features")
    return recon_all

###############################################################################
# Masked Pretraining Collate
###############################################################################
def masked_collate_fn(batch, mask_prob=0.15, logger=print):
    """
    For self-supervised pretraining:
     - We randomly mask a portion of the features for each sample
     - Return (masked_X, original_X) so we can compute MSE
     - Y is not used for pretraining, but let's just store zeros or ignore it
    """
    Xs, Ys = zip(*batch)
    Xs = torch.stack(Xs, dim=0)  # [B, F]
    # Mask
    mask = (torch.rand_like(Xs) < mask_prob)  # boolean of same shape
    # We'll set masked positions to 0, you might choose some sentinel
    masked_X = Xs.clone()
    masked_X[mask] = 0.0
    return masked_X, Xs, mask  # we'll compute MSE on masked positions

###############################################################################
# Fine-Tuning Collate
###############################################################################
def classification_collate_fn(batch):
    Xs, Ys = zip(*batch)
    return torch.stack(Xs, dim=0), torch.tensor(Ys, dtype=torch.long)


###############################################################################
# The Transformer with:
#  - chunking
#  - mixture-of-experts (MoE)
#  - adaptive computation time (ACT)
#  - masked pretraining
###############################################################################

class DynamicLinear(nn.Module):
    """Linear layer wrapper kept for compatibility; not a placeholder."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        self.scaling = alpha / max(1, r)

    def forward(self, x):
        base = x @ self.weight.t() + self.bias
        lora = (x @ self.A.t()) @ self.B.t() * self.scaling
        return base + lora

def alibi_slopes(n_heads):
    slopes = []
    base = 1.0
    for _ in range(n_heads):
        slopes.append(base)
        base *= 0.5
    return torch.tensor(slopes)

class ALiBiMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = LoRALinear(embed_dim, embed_dim)
        self.k_proj = LoRALinear(embed_dim, embed_dim)
        self.v_proj = LoRALinear(embed_dim, embed_dim)
        self.o_proj = LoRALinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=False)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        pos_ids = torch.arange(L, device=x.device).view(1, 1, L)
        rel = pos_ids - pos_ids.transpose(-1, -2)
        slopes_2d = self.slopes.view(self.num_heads, 1, 1).to(x.device)
        alibi = slopes_2d * rel
        alibi = alibi.unsqueeze(0)
        scores = scores + alibi
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.embed_dim)
        out = self.o_proj(out)
        return x + self.rezero * out

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts feed-forward network with E experts. Gate is a small linear -> softmax.
    Each expert is a 2-layer MLP (GELU).
    """
    def __init__(self, d_model, d_ff, E=4, dropout=0.1, top_k=2):
        super().__init__()
        self.E = E
        self.top_k = min(max(1, top_k), E)
        self.gate = nn.Linear(d_model, E)
        self.experts = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = d_ff
        for _ in range(E):
            # each expert is linear -> GELU -> dropout -> linear
            expert = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout)
            )
            self.experts.append(expert)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape
        # gating
        gate_logits = self.gate(x)  # [B, L, E]
        gate_scores = torch.softmax(gate_logits, dim=-1)  # [B, L, E]
        # sparsify: keep top_k experts per token
        if self.top_k < self.E:
            topk_vals, topk_idx = torch.topk(gate_scores, k=self.top_k, dim=-1)
            mask = torch.zeros_like(gate_scores)
            mask.scatter_(-1, topk_idx, 1.0)
            gate_scores = gate_scores * mask
            # renormalize
            denom = gate_scores.sum(dim=-1, keepdim=True) + 1e-9
            gate_scores = gate_scores / denom

        # for each expert, we compute that expert's output
        # then sum up with the gating as weights
        # shape: each expert output => [B, L, d_model]
        out_stack = []
        for e_idx in range(self.E):
            e_out = self.experts[e_idx](x)  # [B, L, d_model]
            # weighting
            w = gate_scores[:,:,e_idx].unsqueeze(-1)  # [B, L, 1]
            e_weighted = e_out * w
            out_stack.append(e_weighted)

        out = sum(out_stack)  # [B, L, d_model]
        return out

class TransformerBlock(nn.Module):
    """
    One encoder block with:
      - multihead attn
      - residual+norm
      - MoE feedforward
      - residual+norm
    """
    def __init__(self, d_model, num_heads, d_ff, E=4, dropout=0.1):
        super().__init__()
        self.mha = ALiBiMultiheadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.moe_ff = MoEFeedForward(d_model, d_ff, E=E, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.rezero_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, L, d_model]
        attn_out = self.ln1(self.mha(x))
        ff_out = self.moe_ff(attn_out)
        x = attn_out + self.rezero_ffn * ff_out
        x = self.ln2(x)
        return x

class AdaptiveComputationTime(nn.Module):
    """
    Wrap the entire stack of blocks. We'll do multiple passes if needed.
    p = sigma(w^T * x_mean)
    If p < threshold, do next pass, up to max_passes.
    We'll keep it simpler: we compute a single pass, then we do p if we do a second pass, etc.
    """
    def __init__(self, d_model, max_passes=3, act_threshold=0.99):
        super().__init__()
        self.halt_linear = nn.Linear(d_model, 1)
        self.max_passes = max_passes
        self.act_threshold = act_threshold

    def forward(self, x, blocks):
        """
        Per-sample ACT with simple halting:
        - After each full pass, compute p_i for each sample
        - Samples with p_i >= threshold are considered halted and keep their last state
        - Continue up to max_passes for remaining samples
        Also computes a differentiable expected pass count for regularization.
        """
        B = x.size(0)
        active = torch.ones(B, 1, device=x.device)
        remainders = torch.ones(B, 1, device=x.device)  # product of (1-p)
        expected_passes = 0.0
        x_current = x
        for t in range(self.max_passes):
            x_next = x_current
            for block in blocks:
                x_next = block(x_next)
            x_mean = x_next.mean(dim=1)
            p = torch.sigmoid(self.halt_linear(x_mean)).unsqueeze(-1)  # [B,1,1] after unsqueeze
            p = p.squeeze(-1)  # [B,1]
            # Expected pass contribution: (t+1) * p * remainders
            expected_passes = expected_passes + (t + 1) * (p * remainders)
            # Determine which samples halt this step
            halt_mask = (p >= self.act_threshold).float()  # [B,1]
            # Update x only for still-active samples
            active_mask = (active > 0.5).float()
            update_mask = (active_mask * (1.0 - halt_mask)).view(B, 1, 1)
            keep_mask = (1.0 - update_mask)
            x_current = x_next * update_mask + x_current * keep_mask
            # Update active and remainders
            active = active * (1.0 - halt_mask)
            remainders = remainders * (1.0 - p)
            # If all halted, stop early
            if active.sum().item() == 0:
                break
        # Save differentiable scalar regularizer
        self.last_expected_passes_tensor = expected_passes.mean()
        return x_current

class AdvancedTransformerClassifier(nn.Module):
    """
    Full pipeline:
     - chunking (embedding)
     - N= e.g. 4 or 6 blocks with multihead attn + MoE
     - ACT gating
     - classification head
    plus a separate 'mask_recon_head' for pretraining
    """
    def __init__(self, input_dim=1280,
                 chunk_len=320,
                 d_model=64,
                 num_heads=4,
                 d_ff=256,
                 E=4,
                 dropout=0.2,
                 n_layers=6,
                 n_classes=3,
                 max_passes=3,
                 act_threshold=0.99,
                 quick_test=False,
                 logger=print):
        super().__init__()
        self.logger = logger
        self.input_dim = input_dim
        self.chunk_len = chunk_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.quick_test = quick_test

        # possibly scale down if quick_test
        if self.quick_test:
            # keep the same dims per paper, but reduce layers for speed in quick mode
            self.n_layers = 2
            self.logger("[INFO] quick_test => n_layers=2 (dims fixed to match paper)")

        # compute how many chunks
        self.num_chunks = (self.input_dim + self.chunk_len - 1)// self.chunk_len

        # input proj
        self.embedding = LoRALinear(self.chunk_len, self.d_model)
        # learned positional encodings per token (num_chunks)
        self.positional = nn.Parameter(torch.zeros(1, (self.input_dim + self.chunk_len - 1)// self.chunk_len, self.d_model))
        nn.init.normal_(self.positional, mean=0.0, std=0.02)

        # Build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads,
                             self.d_ff, E=E, dropout=dropout)
            for _ in range(self.n_layers)
        ])

        # ACT
        self.act = AdaptiveComputationTime(self.d_model, max_passes, act_threshold)

        # RNN integration block per paper table
        self.rnn = nn.GRU(self.d_model, self.d_model // 2, batch_first=True, bidirectional=True)
        self.rnn_proj = nn.Linear(self.d_model, self.d_model)

        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model//2, n_classes)
        )

        # for masked pretraining, a recon head
        # we want to reconstruct chunk_len dims per chunk
        self.mask_recon_head = nn.Linear(self.d_model, self.chunk_len)

    def forward(self, x):
        """
        x shape: [B, F] (F = input_dim). We'll chunk it => [B, num_chunks, chunk_len].
        Then embed => [B, num_chunks, d_model].
        Then pass through ACT + blocks => final => classifier
        We'll do training mode or something. We'll see how to handle. Typically we won't do partial for pretraining. We'll do a separate function for that.
        """
        B, F = x.shape
        # chunk
        # pad if needed
        if F < self.num_chunks*self.chunk_len:
            pad_len = self.num_chunks*self.chunk_len - F
            x = torch.cat([x, torch.zeros(B, pad_len, device=x.device)], dim=1)
        # reshape
        x = x.view(B, self.num_chunks, self.chunk_len)
        # embed each chunk => [B, num_chunks, d_model]
        x = self.embedding(x)
        # add learned positional encodings
        pos = self.positional[:, :self.num_chunks, :]
        x = x + pos
        x = self.act(x, self.blocks)
        # RNN integration
        rnn_out, _ = self.rnn(x)
        x = self.rnn_proj(rnn_out)
        # mean pool across chunks
        x_mean = x.mean(dim=1) # [B, d_model]
        logits = self.classifier(x_mean)
        return logits

    def reconstruct_masked(self, x, mask):
        """
        For pretraining: x shape: [B, F], mask shape: [B, F]
         - we chunk x, embed => pass through blocks => no classification, but we apply recon head chunk by chunk
         - We'll compute MSE on the masked positions
        """
        B, F = x.shape
        # pad if needed
        if F < self.num_chunks*self.chunk_len:
            pad_len = self.num_chunks*self.chunk_len - F
            x = torch.cat([x, torch.zeros(B, pad_len, device=x.device)], dim=1)
            mask = torch.cat([mask, torch.zeros(B, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)

        x_c = x.view(B, self.num_chunks, self.chunk_len)
        emb = self.embedding(x_c)
        # pass through blocks, but let's skip the ACT to keep it simpler in pretraining
        for block in self.blocks:
            emb = block(emb)
        # now apply recon head chunk by chunk
        recon_c = self.mask_recon_head(emb) # [B, num_chunks, chunk_len]
        recon = recon_c.view(B, self.num_chunks*self.chunk_len)
        # if we padded, slice back
        recon = recon[:, :F]
        mask = mask[:, :F]
        return recon, mask


###############################################################################
# Full Train Script
###############################################################################
def train_transformer(args):
    logger = print
    logger("[INFO] Starting train_transformer with args:")
    logger(args)

    # 1) Load dataset
    dataset = MethylationCSVDataset(args.csv, quick_test=args.quick_test, logger=logger)
    n_samples, n_features = dataset.X.shape
    classes = dataset.classes
    logger(f"[DATA] #samples={n_samples}, #features={n_features}, #classes={len(classes)} => {classes}")

    # Per paper: 20% independent hold-out, plus 10-fold stratified CV on train
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.y)
    train_data = torch.utils.data.Subset(dataset, train_idx)
    test_data  = torch.utils.data.Subset(dataset, test_idx)

    # Another split from train_data => for pretraining vs not? Actually we do pretraining on the entire train_data. Then we do fine-tuning on the same data. Or we can do partial.
    # We'll keep it simple: pretrain on entire train_data. Then fine-tune.

    # Optional autoencoder-derived features (reconstructed features)
    if args.enable_autoencoder:
        logger("[AE] Pretraining autoencoder on training split to derive features...")
        # Build a tensor with all data, but train AE using only training indices
        X_all = dataset.X.clone()
        recon_train = run_autoencoder_pretrain(X_all[train_idx], batch_size=max(32, args.batch_size),
                                               epochs=50 if not args.quick_test else 5,
                                               lr=1e-3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               logger=logger)
        # Use AE on full matrix (inference only) to avoid distribution shift
        with torch.no_grad():
            device_ae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ae = DenoisingAutoencoder(input_dim=X_all.shape[1], latent_dim=128).to(device_ae)
            ae.load_state_dict(torch.load("autoencoder.pth", map_location=device_ae))
            ae.eval()
            recon_full = []
            for i in range(0, X_all.size(0), max(32, args.batch_size)):
                xb = X_all[i:i+max(32, args.batch_size)].to(device_ae)
                recon_full.append(ae(xb).cpu())
            recon_full = torch.cat(recon_full, dim=0)
        dataset.X = recon_full  # replace with AE-derived features (still 1280 dims)
        logger("[AE] Replaced dataset.X with reconstructed (denoised) features")

    # 2) Build model
    # if quick_test => chunk_len=200 or something smaller
    chunk_len = args.chunk_len
    if args.quick_test:
        chunk_len = min(256, chunk_len)
    model = AdvancedTransformerClassifier(input_dim=n_features,
                                          chunk_len=chunk_len,
                                          d_model=args.d_model,
                                          num_heads=args.num_heads,
                                          d_ff=args.d_ff,
                                          E=args.num_experts,
                                          dropout=args.dropout,
                                          n_layers=args.n_layers,
                                          n_classes=len(classes),
                                          max_passes=args.max_passes,
                                          act_threshold=args.act_threshold,
                                          quick_test=args.quick_test,
                                          logger=logger)
    logger("[INFO] Model created. Param count = "
           f"{sum(p.numel() for p in model.parameters())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Pretraining
    if args.enable_pretraining:
        logger("[PRETRAIN] Starting masked pretraining stage ...")
        pt_dataset = torch.utils.data.Subset(dataset, train_idx)  # or entire dataset if you prefer
        pt_loader = DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: masked_collate_fn(b, mask_prob=0.15, logger=logger))
        pt_opt = optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        # We'll do a small # epochs for quick test
        pretrain_epochs = args.pretrain_epochs if not args.quick_test else 2
        for ep in range(1, pretrain_epochs+1):
            model.train()
            epoch_loss = 0.0
            for i, (masked_x, orig_x, mask) in enumerate(pt_loader):
                masked_x = masked_x.to(device)
                orig_x   = orig_x.to(device)
                mask     = mask.to(device)
                pt_opt.zero_grad()
                recon, m = model.reconstruct_masked(masked_x, mask)
                # MSE on masked positions
                # We'll only compute on mask=1
                diff = recon - orig_x
                diff2 = diff * diff
                # mask out
                diff2 = diff2 * mask
                loss = diff2.sum() / (mask.sum() + 1e-7)
                loss.backward()
                pt_opt.step()
                epoch_loss += loss.item()
                if (i+1) % 10 == 0:
                    logger(f"[PRETRAIN] ep={ep} iter={i+1}/{len(pt_loader)} partial_loss={loss.item():.6f}")
            epoch_loss /= len(pt_loader)
            logger(f"[PRETRAIN] Ep {ep}/{pretrain_epochs}, avg_loss={epoch_loss:.6f}")
        logger("[PRETRAIN] Done. Saving pretrained weights as 'pretrain_model.pth' ...")
        torch.save(model.state_dict(), "pretrain_model.pth")

    # 4) Fine-tuning
    logger("[TRAIN] Starting fine-tune classification ...")
    # reload pretrained if we want
    if args.enable_pretraining and os.path.exists("pretrain_model.pth"):
        logger("[TRAIN] Loading pretrain_model.pth for fine-tuning ...")
        sd = torch.load("pretrain_model.pth", map_location=device)
        model.load_state_dict(sd, strict=True)

    ft_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                           collate_fn=classification_collate_fn)
    # 10-fold CV on train split (re-train per fold, using pretrained weights if available)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]
    cv_f1s = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        tr_indices = train_idx[tr]
        va_indices = train_idx[va]
        tr_subset = torch.utils.data.Subset(dataset, tr_indices)
        va_subset = torch.utils.data.Subset(dataset, va_indices)
        tr_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True, collate_fn=classification_collate_fn)
        va_loader = DataLoader(va_subset, batch_size=args.batch_size, shuffle=False, collate_fn=classification_collate_fn)

        # fresh model per fold
        model_cv = AdvancedTransformerClassifier(input_dim=n_features,
                                                 chunk_len=chunk_len,
                                                 d_model=args.d_model,
                                                 num_heads=args.num_heads,
                                                 d_ff=args.d_ff,
                                                 E=args.num_experts,
                                                 dropout=args.dropout,
                                                 n_layers=args.n_layers,
                                                 n_classes=len(classes),
                                                 max_passes=args.max_passes,
                                                 act_threshold=args.act_threshold,
                                                 quick_test=args.quick_test,
                                                 logger=logger).to(device)
        if args.enable_pretraining and os.path.exists("pretrain_model.pth"):
            try:
                model_cv.load_state_dict(torch.load("pretrain_model.pth", map_location=device), strict=False)
            except Exception as e:
                logger(f"[CV fold {fold}] Warning: could not load pretrained weights: {e}")
        opt_cv = optim.Adam(model_cv.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
        total_steps_cv = max(1, len(tr_loader) * max(1, (min(10, args.finetune_epochs) if not args.quick_test else 2)))
        warmup_steps_cv = int(0.1 * total_steps_cv)
        def lr_lambda_cv(step):
            if step < warmup_steps_cv:
                return float(step) / float(max(1, warmup_steps_cv))
            progress = float(step - warmup_steps_cv) / float(max(1, total_steps_cv - warmup_steps_cv))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        sched_cv = torch.optim.lr_scheduler.LambdaLR(opt_cv, lr_lambda=lr_lambda_cv)
        crit_cv = nn.CrossEntropyLoss()

        model_cv.train()
        step_ct = 0
        for ep in range(1, (min(10, args.finetune_epochs) if not args.quick_test else 2) + 1):
            for Xb, Yb in tr_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                opt_cv.zero_grad()
                logits = model_cv(Xb)
                act_penalty = getattr(model_cv.act, 'last_expected_passes_tensor', None)
                if act_penalty is None:
                    loss_cv = crit_cv(logits, Yb)
                else:
                    loss_cv = crit_cv(logits, Yb) + 1e-3 * act_penalty
                loss_cv.backward()
                opt_cv.step()
                sched_cv.step()
                step_ct += 1

        # evaluate fold
        model_cv.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for Xb, Yb in va_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                logits = model_cv(Xb)
                pred = torch.argmax(logits, dim=1)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(Yb.cpu().numpy())
        all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([])
        all_trues = np.concatenate(all_trues) if len(all_trues) else np.array([])
        if all_preds.size and all_trues.size:
            f1v = f1_score(all_trues, all_preds, average='macro')
            cv_f1s.append(f1v)
            logger(f"[CV fold {fold}] Macro-F1={f1v:.4f}")
    if cv_f1s:
        logger(f"[CV] 10-fold Macro-F1 mean={np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    val_loader = None
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=classification_collate_fn)

    ft_opt = optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    # Warmup + cosine scheduler
    total_steps = max(1, len(ft_loader) * max(1, (args.finetune_epochs if not args.quick_test else 3)))
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(ft_opt, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()

    max_epochs = args.finetune_epochs if not args.quick_test else 3
    best_f1 = 0.0
    for ep in range(1, max_epochs+1):
        model.train()
        epoch_loss = 0.0
        for i, (Xb, Yb) in enumerate(ft_loader):
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            ft_opt.zero_grad()
            logits = model(Xb)
            # ACT regularization: expected passes term
            act_penalty = getattr(model.act, 'last_expected_passes_tensor', None)
            if act_penalty is None:
            loss = criterion(logits, Yb)
            else:
                loss = criterion(logits, Yb) + 1e-3 * act_penalty
            loss.backward()
            ft_opt.step()
            scheduler.step()
            epoch_loss += loss.item()
            if (i+1) % 10 == 0:
                logger(f"[TRAIN] ep={ep} iter={i+1}/{len(ft_loader)} partial_loss={loss.item():.6f}")

        epoch_loss /= len(ft_loader)
        # Evaluate quickly on train subset or small subset
        # We do a simple measure: macro-F1 on train
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for Xb, Yb in ft_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                logit = model(Xb)
                pred = torch.argmax(logit, dim=1)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(Yb.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        train_f1 = f1_score(all_trues, all_preds, average='macro')
        logger(f"[TRAIN] Ep {ep}/{max_epochs}, loss={epoch_loss:.6f}, train_F1={train_f1:.4f}")

        # If we had val_loader, we would compute val_f1; here we track train_f1
        if train_f1 > best_f1:
            best_f1 = train_f1
            # save
            torch.save(model.state_dict(), "best_model.pth")
            logger(f"[TRAIN] Ep {ep} => new best F1={train_f1:.4f}, saved model.")
        if (train_f1 > 0.99 and not args.quick_test):
            # assume we've basically converged
            logger(f"[TRAIN] Early stopping, train_f1 > 0.99.")
            break

    logger("[TRAIN] Done fine-tuning. Loading best_model.pth for final test evaluation ...")
    best_sd = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(best_sd)

    # Evaluate on test
    logger("[TEST] Evaluating on hold-out test set ...")
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            logit = model(Xb)
            prob = torch.softmax(logit, dim=1)
            pred = torch.argmax(logit, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            all_trues.append(Yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_trues = np.concatenate(all_trues)
    test_f1 = f1_score(all_trues, all_preds, average='macro')

    # confusion
    cm = confusion_matrix(all_trues, all_preds)
    # macro auc
    # if #classes=3, we do a one-vs-rest approach
    try:
        # one-hot
        from sklearn.preprocessing import label_binarize
        Yb_oh = label_binarize(all_trues, classes=range(len(classes)))
        auc_ovr = roc_auc_score(Yb_oh, all_probs, average='macro', multi_class='ovr')
    except:
        auc_ovr = float('nan')
    logger("[TEST] Confusion Matrix:\n" + str(cm))
    acc = (all_preds==all_trues).mean()*100.0
    logger(f"[TEST] Accuracy={acc:.2f}%, Macro-F1={test_f1:.4f}, Macro-AUC={auc_ovr:.4f}")

    # Permutation testing: 100 shuffles of labels on test set
    logger("[TEST] Permutation testing (100 shuffles of TRAIN labels) ...")
    from sklearn.utils import shuffle as skshuffle
    perm_acc = []
    for _ in range(100):
        # shuffle training labels
        y_perm = skshuffle(dataset.y[train_idx], random_state=None)
        perm_subset = torch.utils.data.TensorDataset(dataset.X[train_idx], torch.from_numpy(y_perm))
        perm_loader = DataLoader(perm_subset, batch_size=args.batch_size, shuffle=True)
        # fresh small model for speed
        model_perm = AdvancedTransformerClassifier(input_dim=n_features,
                                                   chunk_len=chunk_len,
                                                   d_model=args.d_model,
                                                   num_heads=args.num_heads,
                                                   d_ff=args.d_ff,
                                                   E=args.num_experts,
                                                   dropout=args.dropout,
                                                   n_layers=max(2, args.n_layers//2),
                                                   n_classes=len(classes),
                                                   max_passes=args.max_passes,
                                                   act_threshold=args.act_threshold,
                                                   quick_test=True,
                                                   logger=lambda *a, **k: None).to(device)
        opt_perm = optim.Adam(model_perm.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
        crit_perm = nn.CrossEntropyLoss()
        model_perm.train()
        for ep in range(2):
            for Xb, Yb in perm_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                opt_perm.zero_grad()
                logits = model_perm(Xb)
                loss_p = crit_perm(logits, Yb)
                loss_p.backward()
                opt_perm.step()
        # evaluate on same test set
        model_perm.eval()
        preds_perm = []
        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(device)
                logit = model_perm(Xb)
                preds_perm.append(torch.argmax(logit, dim=1).cpu().numpy())
        preds_perm = np.concatenate(preds_perm)
        perm_acc.append((preds_perm == all_trues).mean() * 100.0)
    logger(f"[TEST] Permutation test mean accuracy (chance level estimate): {np.mean(perm_acc):.2f}% ± {np.std(perm_acc):.2f}")

    # We are done
    return

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV with shape [samples x features], last col=Condition.")
    parser.add_argument("--quick_test", type=int, default=0, help="1=small ephemeral run for local debugging, 0=full run.")
    parser.add_argument("--enable_pretraining", action='store_true', help="Whether to do masked pretraining first.")
    parser.add_argument("--enable_autoencoder", action='store_true', help="Use autoencoder-derived reconstructed features as transformer input.")
    parser.add_argument("--pretrain_epochs", type=int, default=30, help="Number of epochs for masked pretraining.")
    parser.add_argument("--finetune_epochs", type=int, default=50, help="Number of epochs for classification.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_len", type=int, default=320, help="Number of features per chunk.")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--max_passes", type=int, default=3, help="ACT max passes.")
    parser.add_argument("--act_threshold", type=float, default=0.99)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for transformer blocks and classifier.")

    args = parser.parse_args()
    train_transformer(args)

if __name__ == "__main__":
    main()
