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

        # (Optional) If your CSV has shape [Features x Samples], you might need to transpose:
        # Uncomment if needed:
        df = df.T  # <--- COMMENT THIS OUT OR UNCOMMENT IF SHAPE IS FLIPPED
        # If you do transpose, confirm "Condition" ends up as a column, etc.

        # Condition as the last column
        cond_values = df["Condition"].values.astype(str)
        self.classes = sorted(np.unique(cond_values))
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.y = np.array([self.class2idx[c] for c in cond_values], dtype=np.int64)

        df_feat = df.drop(columns=["Condition"])
        # Convert to float
        Xmat = df_feat.values
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
    """A linear layer that can be smaller if quick_test. Just a normal linear, but we keep naming from prior examples."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # transpose to [B, num_heads, L, head_dim]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        # scaled dot product
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # [B, num_heads, L, head_dim]

        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        out = self.o_proj(out)
        return out

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts feed-forward network with E experts. Gate is a small linear -> softmax.
    Each expert is a 2-layer MLP (GELU).
    """
    def __init__(self, d_model, d_ff, E=4, dropout=0.1):
        super().__init__()
        self.E = E
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
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.moe_ff = MoEFeedForward(d_model, d_ff, E=E, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, d_model]
        attn_out = self.mha(x)
        x = x + attn_out
        x = self.ln1(x)

        ff_out = self.moe_ff(x)
        x = x + ff_out
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
        x shape: [B, L, d_model].
        blocks: list of TransformerBlock
        We do a pass of all blocks => x,
        then compute halting prob. If < threshold => do another pass, etc.
        We'll do it for the entire batch for simplicity,
        though real ACT can do per-sample. We'll do the simpler version.
        """
        for pass_idx in range(self.max_passes):
            # pass the entire stack
            for block in blocks:
                x = block(x)
            # compute halting prob
            x_mean = x.mean(dim=1)  # [B, d_model]
            p = torch.sigmoid(self.halt_linear(x_mean))
            p_mean = p.mean().item()
            # Logging
            # We'll do a small or partial pass if we want. But let's do a simple approach: if p_mean > threshold => break
            if p_mean > self.act_threshold:
                break
        return x  # final

class AdvancedTransformerClassifier(nn.Module):
    """
    Full pipeline:
     - chunking (embedding)
     - N= e.g. 4 or 6 blocks with multihead attn + MoE
     - ACT gating
     - classification head
    plus a separate 'mask_recon_head' for pretraining
    """
    def __init__(self, input_dim=1280, # number of features
                 chunk_len=320, # how many features per chunk
                 d_model=256,
                 num_heads=4,
                 d_ff=1024,
                 E=4, # number of MoE experts
                 dropout=0.1,
                 n_layers=4,
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
            self.d_model = 64
            self.d_ff = 256
            self.logger("[INFO] quick_test => d_model=64, d_ff=256, n_layers=2")

        # compute how many chunks
        self.num_chunks = (self.input_dim + self.chunk_len - 1)// self.chunk_len

        # input proj
        # We'll do a simple linear that goes from chunk_len -> d_model
        # We'll store it as self.embedding
        self.embedding = nn.Linear(self.chunk_len, self.d_model)

        # Build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads,
                             self.d_ff, E=E, dropout=dropout)
            for _ in range(self.n_layers)
        ])

        # ACT
        self.act = AdaptiveComputationTime(self.d_model, max_passes, act_threshold)

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
        # pass into ACT with the blocks
        x = self.act(x, self.blocks)
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

    # We'll do a small hold-out or user can do CV. For simplicity: 80/20 split
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.y)
    train_data = torch.utils.data.Subset(dataset, train_idx)
    test_data  = torch.utils.data.Subset(dataset, test_idx)

    # Another split from train_data => for pretraining vs not? Actually we do pretraining on the entire train_data. Then we do fine-tuning on the same data. Or we can do partial.
    # We'll keep it simple: pretrain on entire train_data. Then fine-tune.

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
        pt_opt = optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
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
    val_loader = None  # we can do a 10% from train as well for val, but let's keep it simpler
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=classification_collate_fn)

    ft_opt = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
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
            loss = criterion(logits, Yb)
            loss.backward()
            ft_opt.step()
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

        # If we had val_loader, we would do val_f1. For demonstration, let's just track train_f1
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
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="Number of epochs for masked pretraining.")
    parser.add_argument("--finetune_epochs", type=int, default=40, help="Number of epochs for classification.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_len", type=int, default=320, help="Number of features per chunk.")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--max_passes", type=int, default=3, help="ACT max passes.")
    parser.add_argument("--act_threshold", type=float, default=0.99)

    args = parser.parse_args()
    train_transformer(args)

if __name__ == "__main__":
    main()
