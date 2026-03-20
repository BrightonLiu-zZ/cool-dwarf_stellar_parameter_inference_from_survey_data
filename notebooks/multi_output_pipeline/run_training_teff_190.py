"""
Stage 1 — Teff Training Script (190 features)
==============================================
Trains a 4-layer feedforward ANN predicting log10(Teff) using:
  - 171 photometric color indices (same as existing Teff model)
  - 19 absolute magnitudes computed from distance_gaia_pc:
      M_band = A_band − 5 × log10(distance_gaia_pc / 10)

Why absolute magnitudes?
  Photometric colors encode Teff extremely well but carry no luminosity information.
  Absolute magnitudes break the color–luminosity degeneracy and are a pre-requisite
  for the Stage 2 log g model (run_training_logg_twostage.py), which needs the best
  possible Teff prediction as an additional input feature.

Usage:
  python run_training_teff_190.py            # start fresh or resume from checkpoint
  python run_training_teff_190.py --reset    # delete checkpoint and start fresh

  Run this script BEFORE run_training_logg_twostage.py.

Output artifacts:
  models/teff_190/stellar_teff_190feat_best.pth
  models/teff_190/scaler_teff_190.pkl
  results/teff_190/test_metrics.json
  results/teff_190/training_diagnostics.png
  results/teff_190/one_to_one_plot.png
  results/teff_190/residual_plots.png
"""

import argparse
import sys
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
RESULTS_DIR  = PROJECT_ROOT / "results" / "teff_190"
MODELS_DIR   = PROJECT_ROOT / "models"  / "teff_190"
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
MAX_EPOCHS          = 100
EARLY_STOP_PATIENCE = 30
BIN_WIDTH_TEFF      = 150.0
MIN_BIN_SAMPLES     = 100
CHECKPOINT_INTERVAL = 5

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

# 171 photometric color indices
COLOR_COLS = [
    f'COLOR_{sorted_mags[i]}_{sorted_mags[j]}'
    for i in range(len(sorted_mags))
    for j in range(i + 1, len(sorted_mags))
]

# 19 absolute magnitudes: M_band = A_band − 5*log10(d/10)
dist_pc       = df["distance_gaia_pc"].values.astype(np.float64)
dist_modulus  = 5.0 * np.log10(dist_pc / 10.0)          # shape (N,)
ABS_MAG_COLS  = [b.replace("A_", "M_") for b in sorted_mags]
abs_mag_arr   = np.column_stack([
    df[band].values - dist_modulus for band in sorted_mags
]).astype(np.float32)                                     # shape (N, 19)

FEATURE_COLS = COLOR_COLS + ABS_MAG_COLS                 # 171 + 19 = 190

color_arr = df[COLOR_COLS].values.astype(np.float32)
X         = np.hstack([color_arr, abs_mag_arr])           # (N, 190)

y              = np.log10(df["teff"].values).astype(np.float32)   # log10(Teff)
spectral_types = df["spectral_type_group"].values

assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NaN/Inf in features!"
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "NaN/Inf in targets!"
log(f"Features: {X.shape[1]}  ({len(COLOR_COLS)} colors + {len(ABS_MAG_COLS)} abs mags)")
log(f"Target: log10(Teff), range [{y.min():.4f}, {y.max():.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15, same SEED as other pipelines)
# ═══════════════════════════════════════════════════════════════════════════════
X_train, X_temp, y_train, y_temp, st_train, st_temp = train_test_split(
    X, y, spectral_types, test_size=0.30, random_state=SEED, stratify=spectral_types)
X_val, X_test, y_val, y_test, st_val, st_test = train_test_split(
    X_temp, y_temp, st_temp, test_size=0.50, random_state=SEED, stratify=st_temp)

teff_train_raw = (10.0 ** y_train).astype(np.float64)
log(f"Split — Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA AUGMENTATION  (Teff-binned, error-aware oversampling)
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
for b in unique_bins:
    mask = bin_ids == b
    bin_sigma_features[b] = np.std(X_train[mask], axis=0)
    bin_sigma_teff_raw[b] = np.std(teff_train_raw[mask])

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

    noise_X      = rng.normal(0.0, bin_sigma_features[b],
                               size=(deficit, X_train.shape[1])).astype(np.float32)
    noise_teff   = rng.normal(0.0, bin_sigma_teff_raw[b], size=deficit)
    new_teff_raw = np.clip(teff_train_raw[src_idx] + noise_teff, 2000.0, None)
    new_log10    = np.log10(new_teff_raw).astype(np.float32)

    aug_X_list.append(X_train[src_idx] + noise_X)
    aug_y_list.append(new_log10)

n_augmented  = sum(len(a) for a in aug_y_list)
X_train_aug  = np.concatenate([X_train] + aug_X_list)
y_train_aug  = np.concatenate([y_train] + aug_y_list)

shuffle_idx = rng.permutation(len(y_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

log(f"Augmented training size: {len(y_train_aug):,}  (+{n_augmented:,} synthetic samples)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE STANDARDIZATION  (scaler fit on original, un-augmented training set)
# ═══════════════════════════════════════════════════════════════════════════════
log("Fitting StandardScaler on original training data ...")
scaler = StandardScaler()
scaler.fit(X_train_orig)

X_train_scaled = scaler.transform(X_train_aug).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════
class StellarDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(StellarDataset(X_train_scaled, y_train_aug),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(StellarDataset(X_val_scaled,   y_val),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
test_loader  = DataLoader(StellarDataset(X_test_scaled,  y_test),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

log(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL  (4-layer feedforward, 190 → 256 → 128 → 64 → 32 → 1)
# ═══════════════════════════════════════════════════════════════════════════════
class StellarTeffNet190(nn.Module):
    """
    Same architecture as the production Teff model, extended to 190 inputs.
    190 → 256 → 128 → 64 → 32 → 1
    Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout
    """
    def __init__(self, input_dim: int, dropout_h1: float = 0.15, dropout_h2: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_h2),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout_h2),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)   # (B,)

model     = StellarTeffNet190(input_dim=len(FEATURE_COLS)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

total_params = sum(p.numel() for p in model.parameters())
log(f"Model: StellarTeffNet190 | Params: {total_params:,} | Input: {len(FEATURE_COLS)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                    patience_counter, train_losses, val_losses, lr_history):
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_model_state": best_model_state,
        "best_val_loss":    best_val_loss,
        "best_epoch":       best_epoch,
        "patience_counter": patience_counter,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "lr_history":       lr_history,
    }, CKPT_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP  (with checkpoint save + resume)
# ═══════════════════════════════════════════════════════════════════════════════
train_losses, val_losses = [], []
lr_history               = []
best_val_loss    = float("inf")
best_epoch       = 0
patience_counter = 0
best_model_state = None
start_epoch      = 1

if CKPT_PATH.exists() and not args.reset:
    log(f"Found checkpoint — resuming from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch      = ckpt["epoch"] + 1
    best_val_loss    = ckpt["best_val_loss"]
    best_epoch       = ckpt["best_epoch"]
    patience_counter = ckpt["patience_counter"]
    best_model_state = ckpt["best_model_state"]
    train_losses     = ckpt["train_losses"]
    val_losses       = ckpt["val_losses"]
    lr_history       = ckpt["lr_history"]
    log(f"Resumed from epoch {ckpt['epoch']}  (best val loss = {best_val_loss:.6f} at epoch {best_epoch})")
else:
    log("Starting fresh training run.")

def train_one_epoch():
    model.train()
    total_loss, n = 0.0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        n          += X_b.size(0)
    return total_loss / n

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        total_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        n          += X_b.size(0)
    return total_loss / n

hdr = f"{'Epoch':>5}  {'TrLoss':>12}  {'VaLoss':>12}  {'LR':>10}  Status"
print(hdr);  print("-" * len(hdr))

stopped_early = False
try:
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0       = time.time()
        tr_loss  = train_one_epoch()
        va_loss  = evaluate(val_loader)
        elapsed  = time.time() - t0
        cur_lr   = optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
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
            print(f"{epoch:>5}  {tr_loss:>12.8f}  {va_loss:>12.8f}  {cur_lr:>10.2e}  {status}  [{elapsed:.1f}s]",
                  flush=True)

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                            patience_counter, train_losses, val_losses, lr_history)
            log(f"Checkpoint saved at epoch {epoch}.")

        if patience_counter >= EARLY_STOP_PATIENCE:
            log(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            stopped_early = True
            break

except KeyboardInterrupt:
    log("KeyboardInterrupt — saving best model so far and continuing to artifacts ...")

if best_model_state is None:
    log("WARNING: no best_model_state saved yet — using current weights.")
else:
    model.load_state_dict(best_model_state)
    log(f"Restored best model from epoch {best_epoch}  (val loss = {best_val_loss:.8f})")

if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    log("Training complete — checkpoint deleted.")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TRAINING DIAGNOSTICS PLOT
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving training diagnostics plot ...")
erange = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
ax.plot(erange, train_losses, label="Train", lw=2)
ax.plot(erange, val_losses,   label="Val",   lw=2)
ax.axvline(best_epoch, color="red", ls="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
ax.set_title("MSE Loss — log10(Teff)"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(erange, lr_history, lw=2, color="green")
ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
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
        all_preds.append(model(X_b.to(DEVICE)).cpu().numpy())
        all_targets.append(y_b.numpy())

log10_teff_pred = np.concatenate(all_preds)
log10_teff_true = np.concatenate(all_targets)
teff_pred_K     = 10.0 ** log10_teff_pred
teff_true_K     = 10.0 ** log10_teff_true

rmse_teff     = np.sqrt(mean_squared_error(teff_true_K, teff_pred_K))
mae_teff      = mean_absolute_error(teff_true_K, teff_pred_K)
r2_teff       = r2_score(teff_true_K, teff_pred_K)
r2_log10_teff = r2_score(log10_teff_true, log10_teff_pred)

print("\n" + "=" * 60)
print("  TEST SET RESULTS — Teff ANN (190 features)")
print("=" * 60)
print(f"  RMSE:       {rmse_teff:.2f} K")
print(f"  MAE:        {mae_teff:.2f} K")
print(f"  R² (K):     {r2_teff:.5f}")
print(f"  R² (log10): {r2_log10_teff:.5f}")

per_type_metrics = {}
print(f"\n  {'Type':<4} {'N':>8}  {'Teff R²':>10}  {'Teff RMSE (K)':>14}")
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    if mask.sum() == 0: continue
    tr2   = r2_score(teff_true_K[mask], teff_pred_K[mask])
    trmse = np.sqrt(mean_squared_error(teff_true_K[mask], teff_pred_K[mask]))
    per_type_metrics[stype] = {"teff_r2": tr2, "teff_rmse_K": trmse}
    print(f"  {stype:<4} {mask.sum():>8,}  {tr2:>10.4f}  {trmse:>14.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving one-to-one plot ...")
fig, ax = plt.subplots(figsize=(9, 8))
hb = ax.hexbin(teff_true_K, teff_pred_K, gridsize=200, cmap="inferno", mincnt=1, bins="log")
plt.colorbar(hb, ax=ax, label="log10(count)")
lims = [min(teff_true_K.min(), teff_pred_K.min()) - 100,
        max(teff_true_K.max(), teff_pred_K.max()) + 100]
ax.plot(lims, lims, "r--", lw=2, alpha=0.8, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("True Teff (K)"); ax.set_ylabel("Predicted Teff (K)")
ax.set_title("Teff — 190-feature ANN (Stage 1)")
ax.text(0.97, 0.03,
        f"RMSE = {rmse_teff:.1f} K\nMAE = {mae_teff:.1f} K\nR² = {r2_teff:.5f}",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "one_to_one_plot.png", dpi=200, bbox_inches="tight")
plt.close()

log("Saving residual plots ...")
res_teff   = teff_pred_K - teff_true_K
colors_map = {"F": "#1f77b4", "G": "#2ca02c", "K": "#ff7f0e", "M": "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
ax = axes[0]
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    ax.scatter(teff_true_K[mask], res_teff[mask], alpha=0.05, s=1,
               color=colors_map[stype], label=stype, rasterized=True)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("True Teff (K)"); ax.set_ylabel("Residual (K)")
ax.set_title("Teff Residuals vs True Teff"); ax.legend(markerscale=20); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(res_teff, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", ls="--", lw=1.5)
ax.axvline(np.mean(res_teff), color="orange", ls="--", lw=1.5,
           label=f"Mean = {np.mean(res_teff):.1f} K")
ax.set_xlabel("Residual (K)"); ax.set_title("Teff Residual Distribution")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "residual_plots.png", dpi=200, bbox_inches="tight")
plt.close()
log(f"Saved residual plots.")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. SAVE FINAL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving model and metrics ...")

torch.save({
    "model_state_dict": best_model_state,
    "input_dim":        len(FEATURE_COLS),
    "feature_cols":     FEATURE_COLS,
    "color_cols":       COLOR_COLS,
    "abs_mag_cols":     ABS_MAG_COLS,
    "sorted_mags":      sorted_mags,
    "best_epoch":       best_epoch,
    "best_val_loss":    best_val_loss,
    "output":           "log10_teff",
}, MODELS_DIR / "stellar_teff_190feat_best.pth")

joblib.dump(scaler, MODELS_DIR / "scaler_teff_190.pkl")

metrics = {
    "model":   "teff_190_ann",
    "teff": {
        "rmse_K":   round(float(rmse_teff), 3),
        "mae_K":    round(float(mae_teff),  3),
        "r2_K":     round(float(r2_teff),   6),
        "r2_log10": round(float(r2_log10_teff), 6),
    },
    "training": {
        "best_epoch":          best_epoch,
        "total_epochs_run":    len(train_losses),
        "n_train_augmented":   len(y_train_aug),
        "n_val":               len(y_val),
        "n_test":              len(y_test),
        "n_features":          len(FEATURE_COLS),
        "n_color_features":    len(COLOR_COLS),
        "n_absmag_features":   len(ABS_MAG_COLS),
        "augmentation_bin_width_K": BIN_WIDTH_TEFF,
    },
    "per_type": {
        k: {"teff_r2": round(v["teff_r2"], 6), "teff_rmse_K": round(v["teff_rmse_K"], 3)}
        for k, v in per_type_metrics.items()
    },
    "baseline_comparison": {
        "teff_171feat_r2":   0.96500,
        "teff_171feat_rmse": 144.0,
        "this_r2":           round(float(r2_teff), 5),
        "this_rmse":         round(float(rmse_teff), 3),
        "delta_r2":          round(float(r2_teff) - 0.96500, 5),
        "delta_rmse":        round(float(rmse_teff) - 144.0, 3),
    },
}
with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("  COMPARISON — Teff")
print("=" * 60)
print(f"  171-feature ANN (baseline):  R²=0.96500  RMSE=144.0 K")
flag = "IMPROVED" if r2_teff > 0.96500 else "REGRESSED"
print(f"  190-feature ANN (this):      R²={r2_teff:.5f}  RMSE={rmse_teff:.1f} K"
      f"  ({r2_teff-0.96500:+.5f} R², {rmse_teff-144.0:+.1f} K)  {flag}")
print(f"\n  Artifacts saved to: {MODELS_DIR} and {RESULTS_DIR}")

log("All done. Run run_training_logg_twostage.py next.")
