#!/usr/bin/env python3
"""
epigenomic_transformer.py

Piyush Acharya, Derek Jacoby
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import matplotlib.pyplot as plt
import wandb  # Added wandb import

# Global hyperparameters
# Per paper defaults
CHUNK_SIZE = 320
BATCH_SIZE = 16
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 50
LEARNING_RATE = 5e-5

MODEL_DIM = 64
FF_DIM    = 256
N_LAYERS  = 6
NUM_HEADS = 4
DROPOUT   = 0.2
MOE_EXPERTS = 4
LORA_R     = 8

# ----------------------------
# Dataset
# ----------------------------
class MethylationChunkedDataset(Dataset):
    """
    Loads a CSV with row=sample, columns=features + Condition.
    Splits each sample's feature vector into chunked tokens of size CHUNK_SIZE.
    """
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        if "Condition" not in df.columns:
            raise ValueError("No 'Condition' column found.")
        self.classes = sorted(df["Condition"].unique())
        self.class_map = {c: i for i, c in enumerate(self.classes)}
        self.labels = np.array([self.class_map[c] for c in df["Condition"].values], dtype=np.int64)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not len(numeric_cols):
            raise ValueError("No numeric features found in CSV.")
        self.data = df[numeric_cols].values.astype(np.float32)
        self.num_features = self.data.shape[1]
        self.data_chunks = []
        for row in self.data:
            tokens = []
            for i in range(0, self.num_features, CHUNK_SIZE):
                tokens.append(row[i:i+CHUNK_SIZE])
            self.data_chunks.append(tokens)
    
    def __len__(self):
        return len(self.data_chunks)
    
    def __getitem__(self, idx):
        return self.data_chunks[idx], self.labels[idx]

def methylation_collate_fn(batch):
    sequences, labels = zip(*batch)
    seq_lengths = [len(seq) for seq in sequences]
    max_seq = max(seq_lengths)
    padded = []
    for seq in sequences:
        tokens = []
        for t in seq:
            t = torch.tensor(t, dtype=torch.float32)
            if len(t) < CHUNK_SIZE:
                pad_len = CHUNK_SIZE - len(t)
                t = torch.cat([t, torch.zeros(pad_len, dtype=torch.float32)], dim=0)
            tokens.append(t)
        while len(tokens) < max_seq:
            tokens.append(torch.zeros(CHUNK_SIZE))
        padded.append(torch.stack(tokens))
    X = torch.stack(padded)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

# ----------------------------
# LoRA Mixin
# ----------------------------
class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation.
    """
    def __init__(self, in_features, out_features, r=LORA_R, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features)*0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=np.sqrt(5))

    def forward(self, x):
        base_out = torch.matmul(x, self.weight.T) + self.bias
        lora_out = torch.matmul(x, self.A.T)
        lora_out = torch.matmul(lora_out, self.B.T) * (self.alpha / self.r)
        return base_out + lora_out

# ----------------------------
# ALiBi Multihead Attention with ReZero
# ----------------------------
def alibi_slopes(n_heads):
    slopes = []
    base = 1.0
    for i in range(n_heads):
        slopes.append(base)
        base *= 0.5
    return torch.tensor(slopes)

class ALiBiMultiheadAttention(nn.Module):
    """Multihead attention with ALiBi bias & ReZero scaling."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.q_proj = LoRALinear(embed_dim, embed_dim)
        self.k_proj = LoRALinear(embed_dim, embed_dim)
        self.v_proj = LoRALinear(embed_dim, embed_dim)
        self.out_proj = LoRALinear(embed_dim, embed_dim)
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=False)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        Q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)
        V = V.permute(0,2,1,3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        pos_ids = torch.arange(seq_len, device=x.device).view(1,1,seq_len)
        pos_j = pos_ids - pos_ids.transpose(-1,-2)
        slopes_2d = self.slopes.view(self.num_heads, 1, 1).to(x.device)
        alibi = slopes_2d * pos_j
        alibi = alibi.unsqueeze(0)
        scores = scores + alibi
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.permute(0,2,1,3).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return x + self.rezero * out

class ALiBiTransformerBlock(nn.Module):
    """
    A single encoder block with ALiBi attention and ReZero-scaled feedforward.
    """
    def __init__(self, embed_dim=MODEL_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.mha = ALiBiMultiheadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            LoRALinear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(ff_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.rezero_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_out = self.ln1(self.mha(x))
        ff_out = self.ffn(attn_out)
        out = attn_out + self.rezero_ffn * ff_out
        out = self.ln2(out)
        return out

# ----------------------------
# RNN Integration and MoE Blocks
# ----------------------------
class RNNIntegrationBlock(nn.Module):
    """Bidirectional GRU integration block"""
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(model_dim, model_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(model_dim, model_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class MoEBlock(nn.Module):
    """Mixture-of-Experts gating block"""
    def __init__(self, model_dim, moe_experts=MOE_EXPERTS):
        super().__init__()
        self.gate = nn.Linear(model_dim, moe_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim)
            ) for _ in range(moe_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_w = torch.softmax(gate_logits, dim=-1)
        expert_outs = [expert(x) for expert in self.experts]
        E = torch.stack(expert_outs, dim=-1)
        gate_w = gate_w.unsqueeze(2)
        out = (E * gate_w).sum(dim=-1)
        return out

# ----------------------------
# Final SOTA Model with Logging for Weight/Bias Convergence
# ----------------------------
class SOTATransformer(nn.Module):
    """
    Full pipeline:
      1) Input projection from CHUNK_SIZE -> MODEL_DIM using LoRA
      2) N_LAYERS of ALiBiTransformerBlock
      3) RNNIntegrationBlock
      4) MoEBlock
      5) Global average pooling
      6) Classifier MLP
    """
    def __init__(self, seq_len, num_classes):
        super().__init__()
        self.seq_len = seq_len
        self.model_dim = MODEL_DIM
        self.input_proj = LoRALinear(CHUNK_SIZE, MODEL_DIM)
        self.blocks = nn.ModuleList([ALiBiTransformerBlock(MODEL_DIM, FF_DIM, NUM_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.rnn_block = RNNIntegrationBlock(MODEL_DIM, DROPOUT)
        self.moe_block = MoEBlock(MODEL_DIM, MOE_EXPERTS)
        self.classifier = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, MODEL_DIM//2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MODEL_DIM//2, num_classes)
        )
    
    def forward(self, x):
        # x: [bsz, seq_len, CHUNK_SIZE]
        bsz, seq_len, _ = x.size()
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.rnn_block(x)
        x = self.moe_block(x)
        x = x.mean(dim=1)
        out = self.classifier(x)
        return out

# ----------------------------
# Functions for Convergence Logging
# ----------------------------
def log_weight_bias_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            norms[name] = param.data.norm().item()
    return norms

def plot_convergence(norms_history, out_path):
    keys = list(norms_history[0].keys())
    epochs = range(1, len(norms_history) + 1)
    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [epoch_dict[key] for epoch_dict in norms_history]
        plt.plot(epochs, values, marker='o', label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Parameter Norm")
    plt.title("Convergence of Transformer Weights and Biases")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Transformer weight/bias convergence graph saved to {out_path}")

# ----------------------------
# Self-Supervised Pretraining with Convergence Logging
# ----------------------------
def self_supervised_pretraining(model, loader, device, epochs=PRETRAIN_EPOCHS, mask_prob=0.15):
    recon_head = LoRALinear(MODEL_DIM, CHUNK_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(recon_head.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    model.train()
    pretrain_loss_history = []
    weight_norms_history = []
    out_dir = "/scratch/derekja/EpiMECoV/results"
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for Xb, _ in loader:
            Xb = Xb.to(device)
            bsz, seq_len, _ = Xb.size()
            mask = (torch.rand(bsz, seq_len, device=device) < mask_prob).float().unsqueeze(-1)
            X_masked = Xb * (1 - mask)
            optimizer.zero_grad()
            emb = model.input_proj(X_masked)
            for blk in model.blocks:
                emb = blk(emb)
            emb = model.rnn_block(emb)
            emb = model.moe_block(emb)
            recon = recon_head(emb)
            loss = criterion(recon * mask, Xb * mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        pretrain_loss_history.append(avg_loss)
        norms = log_weight_bias_norms(model)
        weight_norms_history.append(norms)
        
        # Log to wandb
        wandb.log({
            "pretrain_loss": avg_loss,
            "pretrain_epoch": ep,
            **{f"pretrain_norm_{k}": v for k, v in norms.items()}
        })
        
        logging.info(f"[Pretrain] Epoch {ep}/{epochs}, Loss = {avg_loss:.4f}")
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), pretrain_loss_history, marker='o', label="Pretrain Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Self-Supervised Pretraining Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "pretrain_loss_curve.png"), dpi=300)
    plt.close()
    plot_convergence(weight_norms_history, os.path.join(out_dir, "transformer_weights_convergence.png"))
    logging.info("Self-supervised pretraining complete.")

# ----------------------------
# Finetuning with Convergence Logging
# ----------------------------
def finetune_classifier(model, loader_train, loader_val, device, epochs=FINETUNE_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss_history = []
    val_acc_history = []
    out_dir = "/scratch/derekja/EpiMECoV/results"
    weight_norms_history = []
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for Xb, Yb in loader_train:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, Yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader_train)
        train_loss_history.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xv, Yv in loader_val:
                Xv, Yv = Xv.to(device), Yv.to(device)
                logits = model(Xv)
                _, preds = torch.max(logits, 1)
                correct += (preds == Yv).sum().item()
                total += Yv.size(0)
        acc = correct / total
        val_acc_history.append(acc)
        model.train()
        norms = log_weight_bias_norms(model)
        weight_norms_history.append(norms)
        
        # Log to wandb
        wandb.log({
            "train_loss": avg_loss,
            "val_accuracy": acc,
            "finetune_epoch": ep,
            **{f"finetune_norm_{k}": v for k, v in norms.items()}
        })
        
        logging.info(f"[Finetune] Epoch {ep}/{epochs} => Train Loss = {avg_loss:.4f}, Val Acc = {acc:.4f}")
        plt.figure(figsize=(8,6))
        plt.plot(range(1, ep+1), train_loss_history, marker='o', label="Train Loss")
        plt.plot(range(1, ep+1), val_acc_history, marker='s', label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Finetuning Progress")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "finetune_progress.png"), dpi=300)
        plt.close()
    plot_convergence(weight_norms_history, os.path.join(out_dir, "transformer_weights_convergence_finetune.png"))
    logging.info("Finetuning complete.")

# ----------------------------
# Main Training Pipeline
# ----------------------------
def main():
    logging.basicConfig(filename='results/transformer_classifier.log',level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("entered main")
    # Initialize wandb
    wandb.init(
        project="epimecov-transformer",
        config={
            "chunk_size": CHUNK_SIZE,
            "batch_size": BATCH_SIZE,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_dim": MODEL_DIM,
            "ff_dim": FF_DIM,
            "n_layers": N_LAYERS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "moe_experts": MOE_EXPERTS,
            "lora_r": LORA_R,
            "architecture": "SOTA Transformer with ALiBi + LoRA + MoE"
        }
    )
    logging.info("wandb init")
    
    tmpdir = os.environ['SLURM_TMPDIR']
    logging.info(tmpdir)
    csv_path = tmpdir + "/filtered_biomarker_matrix.csv"
    if not os.path.exists(csv_path):
        logging.error(f"No CSV found: {csv_path}")
        sys.exit(1)
    logging.info("csv")
    dataset = MethylationChunkedDataset(csv_path)
    total = len(dataset)
    logging.info(f"total: {total}")
    train_len = int(0.8 * total)
    val_len = total - train_len
    logging.info(f"val_len: {val_len}")
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=methylation_collate_fn)
    logging.info("train loader")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=methylation_collate_fn)
    logging.info("val_loader")

    seq_len_init = max(len(tokens) for tokens in dataset.data_chunks)
    n_classes = len(dataset.classes)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device for training.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for training.")
    
    logging.info(f"Building SOTA model: seq_len = {seq_len_init}, num_classes = {n_classes}")
    model = SOTATransformer(seq_len_init, n_classes).to(device)
    
    # Log model architecture to wandb
    wandb.watch(model, log="all")
    
    logging.info("==> Starting self-supervised pretraining <==")
    self_supervised_pretraining(model, train_loader, device, epochs=PRETRAIN_EPOCHS, mask_prob=0.15)
    
    logging.info("==> Fine-tuning for classification <==")
    finetune_classifier(model, train_loader, val_loader, device, epochs=FINETUNE_EPOCHS)
    
    out_path = "/scratch/derekja/EpiMECoV/results/transformer_model.pth"
    torch.save(model.state_dict(), out_path)
    logging.info(f"Model saved => {out_path}")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

# Added alias for evaluation compatibility.
TransformerClassifier = SOTATransformer
