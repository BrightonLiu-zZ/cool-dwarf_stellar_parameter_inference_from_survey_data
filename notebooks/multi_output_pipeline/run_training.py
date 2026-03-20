"""
Multi-Output ANN Training Script
=================================
Trains a shared-backbone residual ANN predicting both log10(Teff) and log g simultaneously.

Features:
  - Saves a checkpoint every CHECKPOINT_INTERVAL epochs to models/multi_output/checkpoints/
  - Automatically resumes from the latest checkpoint if one exists
  - Safe to interrupt with Ctrl+C: progress is never more than CHECKPOINT_INTERVAL epochs behind
  - Saves plots as image files (no interactive display needed)

Usage:
  python run_training.py            # start fresh (or resume from checkpoint automatically)
  python run_training.py --reset    # delete existing checkpoint and start fresh
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
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

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true",
                    help="Delete existing checkpoint and train from scratch")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/git_repo/cool-dwarf_stellar_parameter_inference_from_survey_data")
DATA_PATH    = PROJECT_ROOT / "data" / "logg_final_df" / "cool_dwarf_catalog_FGKM_consolidated.csv"
RESULTS_DIR  = PROJECT_ROOT / "results" / "multi_output"
MODELS_DIR   = PROJECT_ROOT / "models"  / "multi_output"
CKPT_DIR     = MODELS_DIR   / "checkpoints"

for d in [RESULTS_DIR, MODELS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CKPT_DIR / "latest_checkpoint.pth"

if args.reset and CKPT_PATH.exists():
    CKPT_PATH.unlink()
    print("Checkpoint deleted — starting fresh.")

# ── Reproducibility & device ──────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE          = 1024
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-5
MAX_EPOCHS          = 100
EARLY_STOP_PATIENCE = 30
BIN_WIDTH_TEFF      = 150.0   # K — Teff-binned augmentation
MIN_BIN_SAMPLES     = 100
CHECKPOINT_INTERVAL = 5       # save checkpoint every N epochs

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
FEATURE_COLS = [
    f'COLOR_{sorted_mags[i]}_{sorted_mags[j]}'
    for i in range(len(sorted_mags))
    for j in range(i + 1, len(sorted_mags))
]

X              = df[FEATURE_COLS].values.astype(np.float32)
log10_teff     = np.log10(df["teff"].values).astype(np.float32)
logg           = df["logg"].values.astype(np.float32)
y              = np.column_stack([log10_teff, logg]).astype(np.float32)
spectral_types = df["spectral_type_group"].values

assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NaN/Inf in features!"
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "NaN/Inf in targets!"
log(f"Features: {X.shape[1]}  |  Targets: {y.shape}")

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
# 3. DATA AUGMENTATION (Teff-binned, error-aware)
# ═══════════════════════════════════════════════════════════════════════════════
log("Running data augmentation ...")
bin_lo    = np.floor(teff_train_raw.min() / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF
bin_hi    = np.ceil(teff_train_raw.max()  / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF + BIN_WIDTH_TEFF
bin_edges = np.arange(bin_lo, bin_hi, BIN_WIDTH_TEFF)
bin_ids   = np.digitize(teff_train_raw, bin_edges)

unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
bins_to_drop = set(unique_bins[bin_counts < MIN_BIN_SAMPLES])
keep_mask    = ~np.isin(bin_ids, list(bins_to_drop))
n_dropped    = (~keep_mask).sum()

X_train       = X_train[keep_mask]
y_train       = y_train[keep_mask]
teff_train_raw = teff_train_raw[keep_mask]
st_train      = st_train[keep_mask]

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
    new_teff_raw   = teff_train_raw[src_idx] + noise_teff
    new_teff_raw   = np.clip(new_teff_raw, 2000.0, None)
    new_log10_teff = np.log10(new_teff_raw).astype(np.float32)
    noise_logg     = rng.normal(0.0, bin_sigma_logg[b], size=deficit)
    new_logg       = np.clip(y_train[src_idx, 1] + noise_logg, 0.0, None).astype(np.float32)

    aug_X_list.append(X_train[src_idx] + noise_X)
    aug_y_list.append(np.column_stack([new_log10_teff, new_logg]).astype(np.float32))

n_augmented  = sum(len(a) for a in aug_y_list)
X_train_aug  = np.concatenate([X_train] + aug_X_list)
y_train_aug  = np.concatenate([y_train] + aug_y_list)
st_train_aug = np.concatenate([st_train] + [st_train[np.where(bin_ids == b)[0][:max_count - c]]
                                             for b, c in zip(unique_bins, bin_counts)
                                             if c < max_count])

shuffle_idx  = rng.permutation(len(y_train_aug))
X_train_aug  = X_train_aug[shuffle_idx]
y_train_aug  = y_train_aug[shuffle_idx]

log(f"Augmented training size: {len(y_train_aug):,}  (+{n_augmented:,} synthetic samples)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
log("Fitting scaler on original (un-augmented) training data ...")
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
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(StellarMultiDataset(X_val_scaled,   y_val),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
test_loader  = DataLoader(StellarMultiDataset(X_test_scaled,  y_test),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

log(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL & LOSS
# ═══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class StellarMultiOutputNet(nn.Module):
    def __init__(self, input_dim, dropout=0.15):
        super().__init__()
        self.backbone = nn.Sequential(ResBlock(input_dim, 256, dropout),
                                      ResBlock(256, 128, dropout))
        self.teff_head = nn.Sequential(nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),
                                       nn.Dropout(0.10), nn.Linear(64,1))
        self.logg_head = nn.Sequential(nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(),
                                       nn.Dropout(0.10), nn.Linear(64,1))
    def forward(self, x):
        shared = self.backbone(x)
        return torch.cat([self.teff_head(shared), self.logg_head(shared)], dim=1)

class HomoscedasticUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_s_teff = nn.Parameter(torch.zeros(1))
        self.log_s_logg = nn.Parameter(torch.zeros(1))
    def forward(self, pred, target):
        mse_teff = F.mse_loss(pred[:, 0], target[:, 0])
        mse_logg = F.mse_loss(pred[:, 1], target[:, 1])
        loss = (torch.exp(-self.log_s_teff) * mse_teff + self.log_s_teff
              + torch.exp(-self.log_s_logg) * mse_logg + self.log_s_logg)
        return loss, mse_teff.detach().item(), mse_logg.detach().item()

model     = StellarMultiOutputNet(input_dim=len(FEATURE_COLS)).to(DEVICE)
criterion = HomoscedasticUncertaintyLoss().to(DEVICE)
optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(epoch, best_model_state, best_crit_state, best_val_loss,
                    best_epoch, patience_counter, train_losses, val_losses,
                    train_mse_teff_hist, val_mse_teff_hist,
                    train_mse_logg_hist, val_mse_logg_hist,
                    log_s_teff_hist, log_s_logg_hist, lr_history):
    torch.save({
        "epoch":               epoch,
        "model_state":         model.state_dict(),
        "criterion_state":     criterion.state_dict(),
        "optimizer_state":     optimizer.state_dict(),
        "scheduler_state":     scheduler.state_dict(),
        "best_model_state":    best_model_state,
        "best_crit_state":     best_crit_state,
        "best_val_loss":       best_val_loss,
        "best_epoch":          best_epoch,
        "patience_counter":    patience_counter,
        "train_losses":        train_losses,
        "val_losses":          val_losses,
        "train_mse_teff_hist": train_mse_teff_hist,
        "val_mse_teff_hist":   val_mse_teff_hist,
        "train_mse_logg_hist": train_mse_logg_hist,
        "val_mse_logg_hist":   val_mse_logg_hist,
        "log_s_teff_hist":     log_s_teff_hist,
        "log_s_logg_hist":     log_s_logg_hist,
        "lr_history":          lr_history,
    }, CKPT_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP (with checkpoint save + resume)
# ═══════════════════════════════════════════════════════════════════════════════
train_losses, val_losses               = [], []
train_mse_teff_hist, val_mse_teff_hist = [], []
train_mse_logg_hist, val_mse_logg_hist = [], []
log_s_teff_hist, log_s_logg_hist       = [], []
lr_history                             = []

best_val_loss    = float("inf")
best_epoch       = 0
patience_counter = 0
best_model_state = None
best_crit_state  = None
start_epoch      = 1

# ── Resume from checkpoint if available ────────────────────────────────────────
if CKPT_PATH.exists() and not args.reset:
    log(f"Found checkpoint at {CKPT_PATH} — resuming ...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    criterion.load_state_dict(ckpt["criterion_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch      = ckpt["epoch"] + 1
    best_val_loss    = ckpt["best_val_loss"]
    best_epoch       = ckpt["best_epoch"]
    patience_counter = ckpt["patience_counter"]
    best_model_state = ckpt["best_model_state"]
    best_crit_state  = ckpt["best_crit_state"]
    train_losses        = ckpt["train_losses"]
    val_losses          = ckpt["val_losses"]
    train_mse_teff_hist = ckpt["train_mse_teff_hist"]
    val_mse_teff_hist   = ckpt["val_mse_teff_hist"]
    train_mse_logg_hist = ckpt["train_mse_logg_hist"]
    val_mse_logg_hist   = ckpt["val_mse_logg_hist"]
    log_s_teff_hist     = ckpt["log_s_teff_hist"]
    log_s_logg_hist     = ckpt["log_s_logg_hist"]
    lr_history          = ckpt["lr_history"]
    log(f"Resumed from epoch {ckpt['epoch']}  (best val loss = {best_val_loss:.6f} at epoch {best_epoch})")
else:
    log("Starting fresh training run.")

def train_one_epoch():
    model.train()
    total_loss = total_mse_t = total_mse_l = 0.0
    n = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_b)
        loss, mse_t, mse_l = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
        bs = X_b.size(0)
        total_loss += loss.item() * bs;  total_mse_t += mse_t * bs;  total_mse_l += mse_l * bs
        n += bs
    return total_loss/n, total_mse_t/n, total_mse_l/n

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = total_mse_t = total_mse_l = 0.0
    n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        preds = model(X_b)
        loss, mse_t, mse_l = criterion(preds, y_b)
        bs = X_b.size(0)
        total_loss += loss.item() * bs;  total_mse_t += mse_t * bs;  total_mse_l += mse_l * bs
        n += bs
    return total_loss/n, total_mse_t/n, total_mse_l/n

hdr = (f"{'Epoch':>5}  {'TrLoss':>10}  {'VaLoss':>10}  "
       f"{'TrMSE_T':>9}  {'VaMSE_T':>9}  {'TrMSE_g':>9}  {'VaMSE_g':>9}  "
       f"{'w_T':>6}  {'w_g':>6}  {'LR':>8}  Status")
print(hdr)
print("-" * len(hdr))

stopped_early = False
for epoch in range(start_epoch, MAX_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_mse_t, tr_mse_l = train_one_epoch()
    va_loss, va_mse_t, va_mse_l = evaluate(val_loader)
    elapsed = time.time() - t0

    current_lr = optimizer.param_groups[0]["lr"]
    ls_t = criterion.log_s_teff.item()
    ls_l = criterion.log_s_logg.item()
    w_t  = float(np.exp(-ls_t))
    w_l  = float(np.exp(-ls_l))

    train_losses.append(tr_loss);         val_losses.append(va_loss)
    train_mse_teff_hist.append(tr_mse_t); val_mse_teff_hist.append(va_mse_t)
    train_mse_logg_hist.append(tr_mse_l); val_mse_logg_hist.append(va_mse_l)
    log_s_teff_hist.append(ls_t);         log_s_logg_hist.append(ls_l)
    lr_history.append(current_lr)

    scheduler.step(va_loss)

    status = ""
    if va_loss < best_val_loss:
        best_val_loss    = va_loss
        best_epoch       = epoch
        patience_counter = 0
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_crit_state  = {k: v.clone() for k, v in criterion.state_dict().items()}
        status = "* Best"
    else:
        patience_counter += 1

    # ── Print row every 5 epochs, on improvement, or on epoch 1 ──────────────
    if epoch % 5 == 0 or status or epoch == start_epoch:
        print(f"{epoch:>5}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
              f"{tr_mse_t:>9.6f}  {va_mse_t:>9.6f}  {tr_mse_l:>9.6f}  {va_mse_l:>9.6f}  "
              f"{w_t:>6.3f}  {w_l:>6.3f}  {current_lr:>8.2e}  {status}  [{elapsed:.1f}s]",
              flush=True)

    # ── Save checkpoint every CHECKPOINT_INTERVAL epochs ─────────────────────
    if epoch % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(epoch, best_model_state, best_crit_state, best_val_loss,
                        best_epoch, patience_counter, train_losses, val_losses,
                        train_mse_teff_hist, val_mse_teff_hist,
                        train_mse_logg_hist, val_mse_logg_hist,
                        log_s_teff_hist, log_s_logg_hist, lr_history)
        log(f"Checkpoint saved at epoch {epoch}.")

    # ── Early stopping ────────────────────────────────────────────────────────
    if patience_counter >= EARLY_STOP_PATIENCE:
        log(f"Early stopping triggered at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
        stopped_early = True
        break

# ── Restore best weights ───────────────────────────────────────────────────────
model.load_state_dict(best_model_state)
criterion.load_state_dict(best_crit_state)
log(f"Restored best model from epoch {best_epoch}  (val loss = {best_val_loss:.6f})")

# ── Delete checkpoint once training is complete (no longer needed) ─────────────
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    log("Training complete — checkpoint deleted.")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TRAINING DIAGNOSTICS PLOT
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving training diagnostics plot ...")
erange = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

ax = axes[0, 0]
ax.plot(erange, train_losses, label="Train", lw=2)
ax.plot(erange, val_losses,   label="Val",   lw=2)
ax.axvline(best_epoch, color="red", ls="--", alpha=0.7, label=f"Best ({best_epoch})")
ax.set_title("Combined Loss (Uncertainty-Weighted)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(erange, train_mse_teff_hist, label="Train Teff",  lw=2)
ax.plot(erange, val_mse_teff_hist,   label="Val Teff",    lw=2)
ax.plot(erange, train_mse_logg_hist, label="Train log g", lw=2, ls="--")
ax.plot(erange, val_mse_logg_hist,   label="Val log g",   lw=2, ls="--")
ax.set_title("Raw MSE per Task"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(erange, [float(np.exp(-s)) for s in log_s_teff_hist], label="Teff weight",  lw=2)
ax.plot(erange, [float(np.exp(-s)) for s in log_s_logg_hist], label="log g weight", lw=2)
ax.set_title("Learned Task Weights"); ax.set_xlabel("Epoch"); ax.set_ylabel("Weight")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(erange, lr_history, lw=2, color="green")
ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
ax.set_yscale("log"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close()
log(f"Saved: {RESULTS_DIR / 'training_diagnostics.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. TEST SET EVALUATION
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

log10_teff_pred = preds_arr[:, 0];    log10_teff_true = targets_arr[:, 0]
teff_pred_K     = 10.0 ** log10_teff_pred;  teff_true_K = 10.0 ** log10_teff_true
logg_pred       = preds_arr[:, 1];    logg_true = targets_arr[:, 1]

rmse_teff     = np.sqrt(mean_squared_error(teff_true_K, teff_pred_K))
mae_teff      = mean_absolute_error(teff_true_K, teff_pred_K)
r2_teff       = r2_score(teff_true_K, teff_pred_K)
r2_log10_teff = r2_score(log10_teff_true, log10_teff_pred)
rmse_logg     = np.sqrt(mean_squared_error(logg_true, logg_pred))
mae_logg      = mean_absolute_error(logg_true, logg_pred)
r2_logg       = r2_score(logg_true, logg_pred)

print("\n" + "=" * 60)
print("  TEST SET RESULTS — Multi-Output ANN")
print("=" * 60)
print(f"  Teff   RMSE={rmse_teff:.2f} K   MAE={mae_teff:.2f} K   R²={r2_teff:.5f}")
print(f"  log g  RMSE={rmse_logg:.4f} dex  MAE={mae_logg:.4f} dex  R²={r2_logg:.5f}")

per_type_metrics = {}
print(f"\n  {'Type':<4} {'N':>8}  {'Teff R²':>9}  {'log g R²':>10}  {'logg RMSE':>10}")
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    if mask.sum() == 0: continue
    tr2  = r2_score(teff_true_K[mask], teff_pred_K[mask])
    lr2  = r2_score(logg_true[mask],   logg_pred[mask])
    lrmse = np.sqrt(mean_squared_error(logg_true[mask], logg_pred[mask]))
    per_type_metrics[stype] = {"teff_r2": tr2, "logg_r2": lr2, "logg_rmse": lrmse}
    print(f"  {stype:<4} {mask.sum():>8,}  {tr2:>9.4f}  {lr2:>10.4f}  {lrmse:>10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PLOTS: ONE-TO-ONE & RESIDUALS
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
ax.set_title("Teff — Multi-Output ANN")
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
ax.set_title("log g — Multi-Output ANN")
ax.text(0.97, 0.03, f"RMSE={rmse_logg:.4f} dex\nMAE={mae_logg:.4f} dex\nR²={r2_logg:.5f}",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "one_to_one_plots.png", dpi=200, bbox_inches="tight")
plt.close()
log(f"Saved: {RESULTS_DIR / 'one_to_one_plots.png'}")

log("Saving residual plots ...")
res_teff = teff_pred_K - teff_true_K
res_logg = logg_pred   - logg_true
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
log(f"Saved: {RESULTS_DIR / 'residual_plots.png'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. SAVE FINAL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving model and metrics ...")

torch.save({
    "model_state_dict":     best_model_state,
    "criterion_state_dict": best_crit_state,
    "input_dim":            len(FEATURE_COLS),
    "feature_cols":         FEATURE_COLS,
    "best_epoch":           best_epoch,
    "best_val_loss":        best_val_loss,
    "outputs":              ["log10_teff", "logg"],
}, MODELS_DIR / "stellar_multi_output_ann_best.pth")

joblib.dump(scaler, MODELS_DIR / "scaler_multi_output.pkl")

metrics = {
    "model": "multi_output_ann",
    "teff":  {"rmse_K": round(float(rmse_teff), 3), "mae_K": round(float(mae_teff), 3),
              "r2_K": round(float(r2_teff), 6), "r2_log10": round(float(r2_log10_teff), 6)},
    "logg":  {"rmse_dex": round(float(rmse_logg), 6), "mae_dex": round(float(mae_logg), 6),
              "r2_score": round(float(r2_logg), 6)},
    "learned_weights": {
        "teff_weight": round(float(np.exp(-criterion.log_s_teff.item())), 6),
        "logg_weight": round(float(np.exp(-criterion.log_s_logg.item())), 6),
    },
    "training": {
        "best_epoch": best_epoch, "total_epochs_run": len(train_losses),
        "n_train_augmented": len(y_train_aug), "n_val": len(y_val), "n_test": len(y_test),
        "n_features": len(FEATURE_COLS), "augmentation_bin_width_K": BIN_WIDTH_TEFF,
    },
    "per_type": {k: {"teff_r2": round(v["teff_r2"], 6), "logg_r2": round(v["logg_r2"], 6),
                     "logg_rmse": round(v["logg_rmse"], 6)} for k, v in per_type_metrics.items()},
}
with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Final comparison summary ───────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  COMPARISON — log g")
print("=" * 68)
baselines = [("4-layer ANN, single-output",  0.487838, 0.20586),
             ("Residual ANN, single-output",  0.519240, 0.19940)]
for name, r2, rmse in baselines:
    print(f"  {name:<38}  R²={r2:.5f}  RMSE={rmse:.5f} dex")
flag = "IMPROVED" if r2_logg > 0.51924 else "REGRESSED"
print(f"  {'Multi-Output ANN (this)':<38}  R²={r2_logg:.5f}  RMSE={rmse_logg:.5f} dex  "
      f"({r2_logg-0.51924:+.5f} R², {rmse_logg-0.19940:+.5f} RMSE)  {flag}")

log("All done.")
