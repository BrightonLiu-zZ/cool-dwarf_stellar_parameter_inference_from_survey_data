"""
Multi-Output ANN with GradNorm  (190 features)
===============================================
Trains a shared-backbone residual ANN predicting both log10(Teff) and log g,
using GradNorm (Chen et al. 2018) for multi-task loss balancing.

Why GradNorm instead of homoscedastic uncertainty weighting?
  The previous run_training.py learned a 194:1 task weight ratio (Teff vs log g),
  effectively ignoring log g. GradNorm actively balances the gradient magnitudes
  across tasks at the last shared backbone layer, preventing one task from drowning out
  the other. It controls task balance via a single hyperparameter α (alpha=1.5 default).

GradNorm mechanics (per training step):
  1. Compute per-task MSE losses: L_teff, L_logg
  2. Compute gradient norms G_i = ||∇_W [w_i * L_i]||₂ at the last backbone layer
  3. Compute target norms: G_target_i = G_mean * (L_i/L0_i / mean(L/L0))^α
  4. GradNorm loss: L_gn = Σ |G_i − G_target_i|  → update task weights only
  5. Weighted task loss: L = Σ w_i * L_i           → update model only
  6. Renormalize: w_i = w_i * n_tasks / Σ w_j

Features vs previous multi-output model:
  - Input: 190 features (171 colors + 19 absolute magnitudes) instead of 171
  - Loss: GradNorm instead of HomoscedasticUncertaintyLoss
  - Two separate Adam optimizers: model params + task weights

Usage:
  python run_training_gradnorm.py            # start fresh or resume
  python run_training_gradnorm.py --reset    # delete checkpoint and restart

Output artifacts:
  models/multi_output_gradnorm/stellar_multi_output_gradnorm_best.pth
  models/multi_output_gradnorm/scaler_gradnorm.pkl
  results/multi_output_gradnorm/test_metrics.json
  results/multi_output_gradnorm/training_diagnostics.png
  results/multi_output_gradnorm/one_to_one_plots.png
  results/multi_output_gradnorm/residual_plots.png
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_context("talk")

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true",
                    help="Delete existing checkpoint and train from scratch")
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/git_repo/cool-dwarf_stellar_parameter_inference_from_survey_data")
DATA_PATH    = PROJECT_ROOT / "data" / "logg_final_df" / "cool_dwarf_catalog_FGKM_consolidated.csv"
RESULTS_DIR  = PROJECT_ROOT / "results" / "multi_output_gradnorm"
MODELS_DIR   = PROJECT_ROOT / "models"  / "multi_output_gradnorm"
CKPT_DIR     = MODELS_DIR   / "checkpoints"

for d in [RESULTS_DIR, MODELS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CKPT_DIR / "latest_checkpoint.pth"

if args.reset and CKPT_PATH.exists():
    CKPT_PATH.unlink()
    print("Checkpoint deleted — starting fresh.")

# ── Reproducibility & device ───────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE          = 1024
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-5
LR_TASK_WEIGHTS     = 1e-3    # separate LR for GradNorm task weights
GRADNORM_ALPHA      = 1.5     # asymmetry parameter; higher = stronger balancing
MAX_EPOCHS          = 100
EARLY_STOP_PATIENCE = 30
BIN_WIDTH_TEFF      = 150.0
MIN_BIN_SAMPLES     = 100
CHECKPOINT_INTERVAL = 5
N_TASKS             = 2

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
log(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
log(f"Dataset shape: {df.shape}")

sorted_mags = [
    'A_BAP', 'A_GSD', 'A_ps_g', 'A_BP', 'A_VAP', 'A_ps_r', 'A_RSD', 'A_RAP',
    'A_GG', 'A_ps_i', 'A_ISD', 'A_RP', 'A_ps_z', 'A_ps_y', 'A_J', 'A_H',
    'A_KS', 'A_W1', 'A_W2'
]
COLOR_COLS   = [
    f'COLOR_{sorted_mags[i]}_{sorted_mags[j]}'
    for i in range(len(sorted_mags))
    for j in range(i + 1, len(sorted_mags))
]
ABS_MAG_COLS = [b.replace("A_", "M_") for b in sorted_mags]

dist_pc     = df["distance_gaia_pc"].values.astype(np.float64)
dist_mod    = 5.0 * np.log10(dist_pc / 10.0)
abs_mag_arr = np.column_stack([df[b].values - dist_mod for b in sorted_mags]).astype(np.float32)

FEATURE_COLS = COLOR_COLS + ABS_MAG_COLS    # 171 + 19 = 190
color_arr    = df[COLOR_COLS].values.astype(np.float32)
X            = np.hstack([color_arr, abs_mag_arr])

log10_teff     = np.log10(df["teff"].values).astype(np.float32)
logg           = df["logg"].values.astype(np.float32)
y              = np.column_stack([log10_teff, logg]).astype(np.float32)
spectral_types = df["spectral_type_group"].values

assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NaN/Inf in features!"
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "NaN/Inf in targets!"
log(f"Features: {X.shape[1]}  ({len(COLOR_COLS)} colors + {len(ABS_MAG_COLS)} abs mags)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_temp, y_train, y_temp, st_train, st_temp = train_test_split(
    X, y, spectral_types, test_size=0.30, random_state=SEED, stratify=spectral_types)
X_val, X_test, y_val, y_test, st_val, st_test = train_test_split(
    X_temp, y_temp, st_temp, test_size=0.50, random_state=SEED, stratify=st_temp)

teff_train_raw = (10.0 ** y_train[:, 0]).astype(np.float64)
log(f"Split — Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA AUGMENTATION  (same Teff-binned strategy as run_training.py)
# ═══════════════════════════════════════════════════════════════════════════════
log("Running Teff-binned data augmentation ...")
bin_lo    = np.floor(teff_train_raw.min() / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF
bin_hi    = np.ceil(teff_train_raw.max()  / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF + BIN_WIDTH_TEFF
bin_edges = np.arange(bin_lo, bin_hi, BIN_WIDTH_TEFF)
bin_ids   = np.digitize(teff_train_raw, bin_edges)

unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
bins_to_drop = set(unique_bins[bin_counts < MIN_BIN_SAMPLES])
keep_mask    = ~np.isin(bin_ids, list(bins_to_drop))

X_train        = X_train[keep_mask]
y_train        = y_train[keep_mask]
teff_train_raw = teff_train_raw[keep_mask]
st_train       = st_train[keep_mask]

bin_ids                 = np.digitize(teff_train_raw, bin_edges)
unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
max_count               = bin_counts.max()

bin_sigma_features = {}
bin_sigma_teff_raw = {}
bin_sigma_logg     = {}
for b in unique_bins:
    mask = bin_ids == b
    bin_sigma_features[b] = np.std(X_train[mask], axis=0)
    bin_sigma_teff_raw[b] = np.std(teff_train_raw[mask])
    bin_sigma_logg[b]     = np.std(y_train[mask, 1])

X_train_orig = X_train.copy()
y_train_orig = y_train.copy()

rng = np.random.default_rng(SEED)
aug_X_list, aug_y_list = [], []

for b, count in zip(unique_bins, bin_counts):
    if count >= max_count:
        continue
    deficit  = max_count - count
    bin_mask = np.where(bin_ids == b)[0]
    src_idx  = rng.choice(bin_mask, size=deficit, replace=True)

    noise_X        = rng.normal(0.0, bin_sigma_features[b],
                                 size=(deficit, X_train.shape[1])).astype(np.float32)
    noise_teff     = rng.normal(0.0, bin_sigma_teff_raw[b], size=deficit)
    new_teff_raw   = np.clip(teff_train_raw[src_idx] + noise_teff, 2000.0, None)
    new_log10_teff = np.log10(new_teff_raw).astype(np.float32)
    noise_logg     = rng.normal(0.0, bin_sigma_logg[b], size=deficit)
    new_logg       = np.clip(y_train[src_idx, 1] + noise_logg, 0.0, None).astype(np.float32)

    aug_X_list.append(X_train[src_idx] + noise_X)
    aug_y_list.append(np.column_stack([new_log10_teff, new_logg]).astype(np.float32))

n_augmented  = sum(len(a) for a in aug_y_list)
X_train_aug  = np.concatenate([X_train] + aug_X_list)
y_train_aug  = np.concatenate([y_train] + aug_y_list)

shuffle_idx = rng.permutation(len(y_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]
log(f"Augmented training size: {len(y_train_aug):,}  (+{n_augmented:,} synthetic samples)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
log("Fitting StandardScaler ...")
scaler = StandardScaler()
scaler.fit(X_train_orig)

X_train_scaled = scaler.transform(X_train_aug).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════
class StellarMultiDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(StellarMultiDataset(X_train_scaled, y_train_aug),
                          batch_size=BATCH_SIZE,   shuffle=True,  num_workers=0)
val_loader   = DataLoader(StellarMultiDataset(X_val_scaled,   y_val),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
test_loader  = DataLoader(StellarMultiDataset(X_test_scaled,  y_test),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
log(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL ARCHITECTURE  (same shared backbone + dual heads as run_training.py)
# ═══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class StellarMultiOutputNet(nn.Module):
    """
    Shared residual backbone + dual output heads.
    Input (190) → ResBlock(256) → ResBlock(128) → [Teff head | log g head] → (B, 2)
    col 0: log10(Teff), col 1: log g
    """
    def __init__(self, input_dim, dropout=0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            ResBlock(input_dim, 256, dropout),
            ResBlock(256, 128, dropout),
        )
        self.teff_head = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                       nn.Dropout(0.10), nn.Linear(64, 1))
        self.logg_head = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                       nn.Dropout(0.10), nn.Linear(64, 1))
    def forward(self, x):
        shared = self.backbone(x)
        return torch.cat([self.teff_head(shared), self.logg_head(shared)], dim=1)

model = StellarMultiOutputNet(input_dim=len(FEATURE_COLS)).to(DEVICE)
log(f"Model: StellarMultiOutputNet | Params: {sum(p.numel() for p in model.parameters()):,} | Input: {len(FEATURE_COLS)}")

# ── GradNorm task weights  (separate from model parameters) ───────────────────
# Initialized to 1.0 for each task; renormalized after each step to sum to N_TASKS
task_weights = nn.Parameter(torch.ones(N_TASKS, device=DEVICE))

# ── Two separate optimizers ───────────────────────────────────────────────────
model_optimizer  = optim.Adam(model.parameters(),   lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
weight_optimizer = optim.Adam([task_weights],        lr=LR_TASK_WEIGHTS)
scheduler        = optim.lr_scheduler.ReduceLROnPlateau(
    model_optimizer, mode="min", factor=0.5, patience=5)

# Reference task losses L0: set on the first training batch, then frozen
# Shape (2,) — L0[0]=initial Teff MSE, L0[1]=initial log g MSE
L0 = None

# Reference to the last linear layer weight in the shared backbone (GradNorm anchor layer)
# backbone[-1] is the second ResBlock; block[4] is its second Linear(128→128)
GRADNORM_LAYER = model.backbone[-1].block[4].weight

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING STEP WITH GRADNORM
# ═══════════════════════════════════════════════════════════════════════════════
def _normalized_weights() -> torch.Tensor:
    """Return task weights clamped positive and normalized to sum to N_TASKS."""
    w = task_weights.clamp(min=1e-4)
    return w * N_TASKS / w.sum()

def train_one_epoch():
    global L0
    model.train()
    total_loss = total_mse_t = total_mse_l = 0.0
    n = 0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        bs = X_b.size(0)

        preds   = model(X_b)
        L_teff  = F.mse_loss(preds[:, 0], y_b[:, 0])
        L_logg  = F.mse_loss(preds[:, 1], y_b[:, 1])
        task_losses = torch.stack([L_teff, L_logg])

        # Initialise reference losses on the very first batch
        if L0 is None:
            L0 = task_losses.detach().clone()
            log(f"GradNorm L0 set: Teff={L0[0]:.6f}  log g={L0[1]:.6f}")

        w = _normalized_weights()

        # ── 1. GradNorm weight update ────────────────────────────────────────
        # Compute per-task gradient norms at the anchor layer (create_graph=True
        # so the norm is differentiable w.r.t. task_weights)
        G_norms = []
        for i in range(N_TASKS):
            g = torch.autograd.grad(
                w[i] * task_losses[i], GRADNORM_LAYER,
                retain_graph=True, create_graph=True,
            )[0]
            G_norms.append(g.norm())

        G_mean      = torch.stack(G_norms).mean().detach()
        loss_ratios = task_losses.detach() / (L0 + 1e-12)
        r_mean      = loss_ratios.mean()
        G_targets   = (G_mean * (loss_ratios / r_mean) ** GRADNORM_ALPHA).detach()
        L_gradnorm  = sum(torch.abs(G_norms[i] - G_targets[i]) for i in range(N_TASKS))

        # Backward through L_gradnorm; clear model param grads so only
        # task_weights get updated by this step
        weight_optimizer.zero_grad()
        model_optimizer.zero_grad()
        L_gradnorm.backward(retain_graph=True)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        weight_optimizer.step()

        # Renormalize weights
        with torch.no_grad():
            task_weights.clamp_(min=1e-4)
            task_weights.mul_(N_TASKS / task_weights.sum())

        # ── 2. Model update ──────────────────────────────────────────────────
        # Use detached (fixed) weights so gradients only flow through task losses
        w_det        = _normalized_weights().detach()
        total_loss_b = (w_det * task_losses).sum()

        model_optimizer.zero_grad()
        total_loss_b.backward()
        model_optimizer.step()

        total_loss += total_loss_b.item() * bs
        total_mse_t += L_teff.detach().item() * bs
        total_mse_l += L_logg.detach().item() * bs
        n += bs

    return total_loss / n, total_mse_t / n, total_mse_l / n

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = total_mse_t = total_mse_l = 0.0
    n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        preds   = model(X_b)
        L_teff  = F.mse_loss(preds[:, 0], y_b[:, 0])
        L_logg  = F.mse_loss(preds[:, 1], y_b[:, 1])
        w_det   = _normalized_weights().detach()
        loss    = (w_det * torch.stack([L_teff, L_logg])).sum()
        bs      = X_b.size(0)
        total_loss += loss.item() * bs
        total_mse_t += L_teff.item() * bs
        total_mse_l += L_logg.item() * bs
        n += bs
    return total_loss / n, total_mse_t / n, total_mse_l / n

# ═══════════════════════════════════════════════════════════════════════════════
# 8. CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                    patience_counter, train_losses, val_losses,
                    train_mse_teff_hist, val_mse_teff_hist,
                    train_mse_logg_hist, val_mse_logg_hist,
                    w_teff_hist, w_logg_hist, lr_history):
    torch.save({
        "epoch":               epoch,
        "model_state":         model.state_dict(),
        "task_weights":        task_weights.detach().clone(),
        "L0":                  L0,
        "model_optim_state":   model_optimizer.state_dict(),
        "weight_optim_state":  weight_optimizer.state_dict(),
        "scheduler_state":     scheduler.state_dict(),
        "best_model_state":    best_model_state,
        "best_val_loss":       best_val_loss,
        "best_epoch":          best_epoch,
        "patience_counter":    patience_counter,
        "train_losses":        train_losses,
        "val_losses":          val_losses,
        "train_mse_teff_hist": train_mse_teff_hist,
        "val_mse_teff_hist":   val_mse_teff_hist,
        "train_mse_logg_hist": train_mse_logg_hist,
        "val_mse_logg_hist":   val_mse_logg_hist,
        "w_teff_hist":         w_teff_hist,
        "w_logg_hist":         w_logg_hist,
        "lr_history":          lr_history,
    }, CKPT_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
train_losses, val_losses               = [], []
train_mse_teff_hist, val_mse_teff_hist = [], []
train_mse_logg_hist, val_mse_logg_hist = [], []
w_teff_hist, w_logg_hist               = [], []
lr_history                             = []
best_val_loss    = float("inf")
best_epoch       = 0
patience_counter = 0
best_model_state = None
start_epoch      = 1

if CKPT_PATH.exists() and not args.reset:
    log(f"Found checkpoint — resuming from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    with torch.no_grad():
        task_weights.copy_(ckpt["task_weights"].to(DEVICE))
    L0 = ckpt["L0"].to(DEVICE) if ckpt["L0"] is not None else None
    model_optimizer.load_state_dict(ckpt["model_optim_state"])
    weight_optimizer.load_state_dict(ckpt["weight_optim_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch          = ckpt["epoch"] + 1
    best_val_loss        = ckpt["best_val_loss"]
    best_epoch           = ckpt["best_epoch"]
    patience_counter     = ckpt["patience_counter"]
    best_model_state     = ckpt["best_model_state"]
    train_losses         = ckpt["train_losses"]
    val_losses           = ckpt["val_losses"]
    train_mse_teff_hist  = ckpt["train_mse_teff_hist"]
    val_mse_teff_hist    = ckpt["val_mse_teff_hist"]
    train_mse_logg_hist  = ckpt["train_mse_logg_hist"]
    val_mse_logg_hist    = ckpt["val_mse_logg_hist"]
    w_teff_hist          = ckpt["w_teff_hist"]
    w_logg_hist          = ckpt["w_logg_hist"]
    lr_history           = ckpt["lr_history"]
    log(f"Resumed from epoch {ckpt['epoch']}  (best val loss = {best_val_loss:.6f})")
else:
    log("Starting fresh training run.")

hdr = (f"{'Epoch':>5}  {'TrLoss':>10}  {'VaLoss':>10}  "
       f"{'TrMSE_T':>9}  {'VaMSE_T':>9}  {'TrMSE_g':>9}  {'VaMSE_g':>9}  "
       f"{'w_T':>6}  {'w_g':>6}  {'LR':>8}  Status")
print(hdr);  print("-" * len(hdr))

try:
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_mse_t, tr_mse_l = train_one_epoch()
        va_loss, va_mse_t, va_mse_l = evaluate(val_loader)
        elapsed = time.time() - t0

        w_now   = _normalized_weights().detach().cpu()
        cur_lr  = model_optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss);         val_losses.append(va_loss)
        train_mse_teff_hist.append(tr_mse_t); val_mse_teff_hist.append(va_mse_t)
        train_mse_logg_hist.append(tr_mse_l); val_mse_logg_hist.append(va_mse_l)
        w_teff_hist.append(float(w_now[0]));  w_logg_hist.append(float(w_now[1]))
        lr_history.append(cur_lr)

        scheduler.step(va_loss)

        status = ""
        if va_loss < best_val_loss:
            best_val_loss    = va_loss
            best_epoch       = epoch
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            status = "* Best"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or status or epoch == start_epoch:
            print(f"{epoch:>5}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                  f"{tr_mse_t:>9.6f}  {va_mse_t:>9.6f}  {tr_mse_l:>9.6f}  {va_mse_l:>9.6f}  "
                  f"{float(w_now[0]):>6.3f}  {float(w_now[1]):>6.3f}  {cur_lr:>8.2e}  "
                  f"{status}  [{elapsed:.1f}s]", flush=True)

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                            patience_counter, train_losses, val_losses,
                            train_mse_teff_hist, val_mse_teff_hist,
                            train_mse_logg_hist, val_mse_logg_hist,
                            w_teff_hist, w_logg_hist, lr_history)
            log(f"Checkpoint saved at epoch {epoch}.")

        if patience_counter >= EARLY_STOP_PATIENCE:
            log(f"Early stopping at epoch {epoch}.")
            break

except KeyboardInterrupt:
    log("KeyboardInterrupt — saving best model so far and continuing to artifacts ...")

if best_model_state is None:
    log("WARNING: no best_model_state saved yet — using current weights.")
else:
    model.load_state_dict(best_model_state)
    log(f"Restored best model from epoch {best_epoch}  (val loss = {best_val_loss:.6f})")
    log(f"Final task weights — Teff: {float(w_now[0]):.4f}  log g: {float(w_now[1]):.4f}")

if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    log("Training complete — checkpoint deleted.")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. TRAINING DIAGNOSTICS PLOT
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving training diagnostics plot ...")
erange = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

ax = axes[0, 0]
ax.plot(erange, train_losses, label="Train", lw=2)
ax.plot(erange, val_losses,   label="Val",   lw=2)
ax.axvline(best_epoch, color="red", ls="--", alpha=0.7, label=f"Best ({best_epoch})")
ax.set_title("Combined Loss (GradNorm-Weighted)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(erange, train_mse_teff_hist, label="Train Teff",  lw=2)
ax.plot(erange, val_mse_teff_hist,   label="Val Teff",    lw=2)
ax.plot(erange, train_mse_logg_hist, label="Train log g", lw=2, ls="--")
ax.plot(erange, val_mse_logg_hist,   label="Val log g",   lw=2, ls="--")
ax.set_title("Raw MSE per Task"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(erange, w_teff_hist, label="Teff weight",  lw=2)
ax.plot(erange, w_logg_hist, label="log g weight", lw=2)
ax.axhline(1.0, color="grey", ls=":", lw=1, alpha=0.6, label="Equal weight (1.0)")
ax.set_title("GradNorm Task Weights (normalized to sum=2)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Weight")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(erange, lr_history, lw=2, color="green")
ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
ax.set_yscale("log"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 11. TEST SET EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
log("Evaluating on test set ...")
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        preds = model(X_b.to(DEVICE))
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_b.numpy())

preds_arr   = np.concatenate(all_preds)
targets_arr = np.concatenate(all_targets)

log10_teff_pred = preds_arr[:, 0];  log10_teff_true = targets_arr[:, 0]
teff_pred_K     = 10.0 ** log10_teff_pred;  teff_true_K = 10.0 ** log10_teff_true
logg_pred       = preds_arr[:, 1];  logg_true = targets_arr[:, 1]

rmse_teff     = np.sqrt(mean_squared_error(teff_true_K, teff_pred_K))
mae_teff      = mean_absolute_error(teff_true_K, teff_pred_K)
r2_teff       = r2_score(teff_true_K, teff_pred_K)
r2_log10_teff = r2_score(log10_teff_true, log10_teff_pred)
rmse_logg     = np.sqrt(mean_squared_error(logg_true, logg_pred))
mae_logg      = mean_absolute_error(logg_true, logg_pred)
r2_logg       = r2_score(logg_true, logg_pred)

print("\n" + "=" * 60)
print("  TEST SET RESULTS — Multi-Output ANN + GradNorm")
print("=" * 60)
print(f"  Teff:  RMSE={rmse_teff:.2f} K   MAE={mae_teff:.2f} K   R²={r2_teff:.5f}")
print(f"  log g: RMSE={rmse_logg:.4f} dex  MAE={mae_logg:.4f} dex  R²={r2_logg:.5f}")

per_type_metrics = {}
print(f"\n  {'Type':<4} {'N':>8}  {'Teff R²':>9}  {'log g R²':>10}  {'logg RMSE':>10}")
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    if mask.sum() == 0: continue
    tr2   = r2_score(teff_true_K[mask], teff_pred_K[mask])
    lr2   = r2_score(logg_true[mask],   logg_pred[mask])
    lrmse = np.sqrt(mean_squared_error(logg_true[mask], logg_pred[mask]))
    per_type_metrics[stype] = {"teff_r2": tr2, "logg_r2": lr2, "logg_rmse": lrmse}
    print(f"  {stype:<4} {mask.sum():>8,}  {tr2:>9.4f}  {lr2:>10.4f}  {lrmse:>10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving one-to-one plots ...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

ax = axes[0]
hb = ax.hexbin(teff_true_K, teff_pred_K, gridsize=200, cmap="inferno", mincnt=1, bins="log")
plt.colorbar(hb, ax=ax, label="log10(count)")
lims = [min(teff_true_K.min(), teff_pred_K.min())-100, max(teff_true_K.max(), teff_pred_K.max())+100]
ax.plot(lims, lims, "r--", lw=2, alpha=0.8)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("True Teff (K)"); ax.set_ylabel("Predicted Teff (K)")
ax.set_title("Teff — Multi-Output GradNorm ANN")
ax.text(0.97, 0.03, f"RMSE={rmse_teff:.1f} K\nMAE={mae_teff:.1f} K\nR²={r2_teff:.5f}",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

ax = axes[1]
hb = ax.hexbin(logg_true, logg_pred, gridsize=200, cmap="inferno", mincnt=1, bins="log")
plt.colorbar(hb, ax=ax, label="log10(count)")
lims = [min(logg_true.min(), logg_pred.min())-0.1, max(logg_true.max(), logg_pred.max())+0.1]
ax.plot(lims, lims, "r--", lw=2, alpha=0.8)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("True log g (dex)"); ax.set_ylabel("Predicted log g (dex)")
ax.set_title("log g — Multi-Output GradNorm ANN")
ax.text(0.97, 0.03, f"RMSE={rmse_logg:.4f} dex\nMAE={mae_logg:.4f} dex\nR²={r2_logg:.5f}",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "one_to_one_plots.png", dpi=200, bbox_inches="tight")
plt.close()

log("Saving residual plots ...")
res_teff   = teff_pred_K - teff_true_K
res_logg   = logg_pred   - logg_true
colors_map = {"F": "#1f77b4", "G": "#2ca02c", "K": "#ff7f0e", "M": "#d62728"}

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
ax = axes[0, 0]
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    ax.scatter(teff_true_K[mask], res_teff[mask], alpha=0.05, s=1,
               color=colors_map[stype], label=stype, rasterized=True)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("True Teff (K)"); ax.set_ylabel("Residual (K)"); ax.set_title("Teff Residuals")
ax.legend(markerscale=20); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(res_teff, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", ls="--", lw=1.5)
ax.axvline(np.mean(res_teff), color="orange", ls="--", lw=1.5,
           label=f"Mean={np.mean(res_teff):.1f} K")
ax.set_xlabel("Residual (K)"); ax.set_title("Teff Residual Distribution"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    ax.scatter(logg_true[mask], res_logg[mask], alpha=0.05, s=1,
               color=colors_map[stype], label=stype, rasterized=True)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("True log g (dex)"); ax.set_ylabel("Residual (dex)"); ax.set_title("log g Residuals")
ax.legend(markerscale=20); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(res_logg, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", ls="--", lw=1.5)
ax.axvline(np.mean(res_logg), color="orange", ls="--", lw=1.5,
           label=f"Mean={np.mean(res_logg):.4f} dex")
ax.set_xlabel("Residual (dex)"); ax.set_title("log g Residual Distribution"); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "residual_plots.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 13. SAVE FINAL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving model and metrics ...")
w_final = _normalized_weights().detach().cpu()

torch.save({
    "model_state_dict": best_model_state,
    "task_weights":     task_weights.detach().cpu().clone(),
    "input_dim":        len(FEATURE_COLS),
    "feature_cols":     FEATURE_COLS,
    "color_cols":       COLOR_COLS,
    "abs_mag_cols":     ABS_MAG_COLS,
    "best_epoch":       best_epoch,
    "best_val_loss":    best_val_loss,
    "outputs":          ["log10_teff", "logg"],
    "gradnorm_alpha":   GRADNORM_ALPHA,
}, MODELS_DIR / "stellar_multi_output_gradnorm_best.pth")

joblib.dump(scaler, MODELS_DIR / "scaler_gradnorm.pkl")

metrics = {
    "model": "multi_output_gradnorm_ann",
    "teff":  {"rmse_K": round(float(rmse_teff), 3), "mae_K": round(float(mae_teff), 3),
               "r2_K": round(float(r2_teff), 6), "r2_log10": round(float(r2_log10_teff), 6)},
    "logg":  {"rmse_dex": round(float(rmse_logg), 6), "mae_dex": round(float(mae_logg), 6),
               "r2_score": round(float(r2_logg), 6)},
    "final_task_weights": {
        "teff_weight": round(float(w_final[0]), 4),
        "logg_weight": round(float(w_final[1]), 4),
    },
    "training": {
        "best_epoch": best_epoch, "total_epochs_run": len(train_losses),
        "n_train_augmented": len(y_train_aug), "n_val": len(y_val), "n_test": len(y_test),
        "n_features": len(FEATURE_COLS), "gradnorm_alpha": GRADNORM_ALPHA,
        "augmentation_bin_width_K": BIN_WIDTH_TEFF,
    },
    "per_type": {k: {"teff_r2": round(v["teff_r2"], 6), "logg_r2": round(v["logg_r2"], 6),
                     "logg_rmse": round(v["logg_rmse"], 6)} for k, v in per_type_metrics.items()},
    "baselines": {
        "residual_single_output": {"r2": 0.51924, "rmse": 0.19940},
        "multi_output_homosc":    {"r2": 0.48343, "rmse": 0.20674},
    },
}
with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 68)
print("  COMPARISON — log g")
print("=" * 68)
print(f"  {'Residual ANN, single-output':<38}  R²=0.51924  RMSE=0.19940 dex")
print(f"  {'Multi-output (homoscedastic)':<38}  R²=0.48343  RMSE=0.20674 dex")
flag = "IMPROVED" if r2_logg > 0.51924 else "REGRESSED"
print(f"  {'Multi-output + GradNorm (this)':<38}  R²={r2_logg:.5f}  RMSE={rmse_logg:.5f} dex"
      f"  ({r2_logg-0.51924:+.5f} R²)  {flag}")
print(f"  Final GradNorm weights — Teff: {float(w_final[0]):.4f}  log g: {float(w_final[1]):.4f}"
      f"  (should be closer to 1:1 than 194:1 from before)")
print(f"\n  Artifacts saved to: {MODELS_DIR} and {RESULTS_DIR}")

log("All done.")
