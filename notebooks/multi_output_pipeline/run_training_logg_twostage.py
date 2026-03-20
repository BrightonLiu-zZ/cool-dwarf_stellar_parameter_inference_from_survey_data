"""
Stage 2 — log g Training Script (Two-Stage Pipeline)
=====================================================
Uses the Stage 1 Teff model (run_training_teff_190.py) to generate log10(Teff)
predictions, then trains a dedicated log g model that takes those predictions as
an additional feature alongside the 190 photometric features.

Input features (191 total):
  - 190 features from Stage 1 (171 color indices + 19 absolute magnitudes)
  - 1 predicted log10(Teff) from the Stage 1 model  ← new signal for log g

Why this helps:
  Teff and log g are correlated along the main sequence (hotter dwarfs tend to have
  lower log g). By feeding in the Stage 1 Teff prediction, the log g model can learn
  this physical relationship and use Teff as a strong prior to narrow down log g.

Prerequisites:
  models/teff_190/stellar_teff_190feat_best.pth  (Stage 1 Teff model)
  models/teff_190/scaler_teff_190.pkl

Usage:
  python run_training_logg_twostage.py            # start fresh or resume
  python run_training_logg_twostage.py --reset    # delete checkpoint and restart

Output artifacts:
  models/logg_twostage/stellar_logg_twostage_best.pth
  models/logg_twostage/scaler_logg_twostage.pkl
  results/logg_twostage/test_metrics.json
  results/logg_twostage/training_diagnostics.png
  results/logg_twostage/one_to_one_plot.png
  results/logg_twostage/residual_plots.png
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

# ── Stage 1 variant ────────────────────────────────────────────────────────────
# "teff_190" → new 190-feature Teff model (run_training_teff_190.py)
# "baseline" → original 171-feature Teff model (models/teff/stellar_teff_ann_best.pth)
STAGE1_VARIANT = "baseline"

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("C:/git_repo/cool-dwarf_stellar_parameter_inference_from_survey_data")
DATA_PATH     = PROJECT_ROOT / "data" / "logg_final_df" / "cool_dwarf_catalog_FGKM_consolidated.csv"

if STAGE1_VARIANT == "teff_190":
    STAGE1_MODEL     = PROJECT_ROOT / "models" / "teff_190" / "stellar_teff_190feat_best.pth"
    STAGE1_SCALER    = PROJECT_ROOT / "models" / "teff_190" / "scaler_teff_190.pkl"
    STAGE1_INPUT_DIM = 190
else:  # "baseline" — original 171-feature model
    STAGE1_MODEL     = PROJECT_ROOT / "models" / "teff"     / "stellar_teff_ann_best.pth"
    STAGE1_SCALER    = PROJECT_ROOT / "models" / "teff"     / "scaler.pkl"
    STAGE1_INPUT_DIM = 171
RESULTS_DIR   = PROJECT_ROOT / "results" / "logg_twostage"
MODELS_DIR    = PROJECT_ROOT / "models"  / "logg_twostage"
CKPT_DIR      = MODELS_DIR   / "checkpoints"

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
MAX_EPOCHS          = 150
EARLY_STOP_PATIENCE = 30
BIN_WIDTH_TEFF      = 150.0
MIN_BIN_SAMPLES     = 100
CHECKPOINT_INTERVAL = 5

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── Check Stage 1 artifacts exist ─────────────────────────────────────────────
if not STAGE1_MODEL.exists():
    print(f"\nERROR: Stage 1 model not found at {STAGE1_MODEL}")
    print("Please run run_training_teff_190.py first.\n")
    sys.exit(1)
if not STAGE1_SCALER.exists():
    print(f"\nERROR: Stage 1 scaler not found at {STAGE1_SCALER}")
    print("Please run run_training_teff_190.py first.\n")
    sys.exit(1)

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

FEATURE_COLS_190 = COLOR_COLS + ABS_MAG_COLS    # 190
color_arr = df[COLOR_COLS].values.astype(np.float32)
X_190     = np.hstack([color_arr, abs_mag_arr])

y_logg         = df["logg"].values.astype(np.float32)
teff_all       = df["teff"].values.astype(np.float32)
spectral_types = df["spectral_type_group"].values

assert not np.any(np.isnan(X_190)) and not np.any(np.isinf(X_190)), "NaN/Inf in 190 features!"
log(f"Base features: 190  |  log g range: [{y_logg.min():.3f}, {y_logg.max():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT  (SEED=42 matches all other pipelines)
# ═══════════════════════════════════════════════════════════════════════════════
(X_train_190, X_temp_190,
 y_train, y_temp,
 st_train, st_temp,
 teff_train_raw, _) = train_test_split(
    X_190, y_logg, spectral_types, teff_all,
    test_size=0.30, random_state=SEED, stratify=spectral_types)

(X_val_190, X_test_190,
 y_val, y_test,
 st_val, st_test) = train_test_split(
    X_temp_190, y_temp, st_temp,
    test_size=0.50, random_state=SEED, stratify=st_temp)

teff_train_raw = teff_train_raw.astype(np.float64)
log(f"Split — Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

# 171-feature (color-only) splits — used when STAGE1_VARIANT == "baseline"
X_train_171 = X_train_190[:, :171]
X_val_171   = X_val_190[:,   :171]
X_test_171  = X_test_190[:,  :171]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. STAGE 1 — Load Teff model & generate predictions for all splits
# ═══════════════════════════════════════════════════════════════════════════════
log(f"Loading Stage 1 Teff model  [variant: {STAGE1_VARIANT}] ...")

class StellarTeffNet190(nn.Module):
    """190-feature Teff model from run_training_teff_190.py  (attr: self.net)."""
    def __init__(self, input_dim, dropout_h1=0.15, dropout_h2=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_h2),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout_h2),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class StellarTeffNetBaseline(nn.Module):
    """Original 171-feature Teff model (attr: self.network, no Dropout before output)."""
    def __init__(self, input_dim=171, dropout_h1=0.15, dropout_h2=0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_h1),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_h2),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

ckpt_s1      = torch.load(STAGE1_MODEL, map_location=DEVICE)
input_dim_s1 = ckpt_s1.get("input_dim", STAGE1_INPUT_DIM)   # baseline may not store input_dim
if STAGE1_VARIANT == "teff_190":
    teff_model = StellarTeffNet190(input_dim=input_dim_s1).to(DEVICE)
else:
    teff_model = StellarTeffNetBaseline(input_dim=input_dim_s1).to(DEVICE)
teff_model.load_state_dict(ckpt_s1["model_state_dict"])
teff_model.eval()
scaler_s1  = joblib.load(STAGE1_SCALER)
log(f"  Stage 1 loaded  (input_dim={input_dim_s1}, best epoch: {ckpt_s1.get('best_epoch', '?')})")

@torch.no_grad()
def predict_log10_teff(X_raw: np.ndarray) -> np.ndarray:
    """Run Stage 1 inference; returns log10(Teff) predictions, shape (N,)."""
    X_sc  = scaler_s1.transform(X_raw).astype(np.float32)
    preds = []
    for i in range(0, len(X_sc), BATCH_SIZE * 4):
        batch = torch.tensor(X_sc[i:i + BATCH_SIZE * 4]).to(DEVICE)
        preds.append(teff_model(batch).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)

# Choose Stage 1 input arrays based on variant
_X_s1_train = X_train_190 if STAGE1_VARIANT == "teff_190" else X_train_171
_X_s1_val   = X_val_190   if STAGE1_VARIANT == "teff_190" else X_val_171
_X_s1_test  = X_test_190  if STAGE1_VARIANT == "teff_190" else X_test_171

log(f"Generating Stage 1 Teff predictions for all splits  [{STAGE1_VARIANT}, {input_dim_s1} features] ...")
teff_pred_train = predict_log10_teff(_X_s1_train)
teff_pred_val   = predict_log10_teff(_X_s1_val)
teff_pred_test  = predict_log10_teff(_X_s1_test)

# Append predicted log10(Teff) as the 191st feature
X_train_191     = np.hstack([X_train_190, teff_pred_train[:, None]])
X_val_191       = np.hstack([X_val_190,   teff_pred_val[:, None]])
X_test_191      = np.hstack([X_test_190,  teff_pred_test[:, None]])
FEATURE_COLS_191 = FEATURE_COLS_190 + ["pred_log10_teff"]
log(f"Feature set: {X_train_191.shape[1]}  (190 photometric + 1 predicted log10 Teff)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA AUGMENTATION  (Teff-binned on 191 features)
# ═══════════════════════════════════════════════════════════════════════════════
log("Running Teff-binned data augmentation ...")
bin_lo    = np.floor(teff_train_raw.min() / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF
bin_hi    = np.ceil(teff_train_raw.max()  / BIN_WIDTH_TEFF) * BIN_WIDTH_TEFF + BIN_WIDTH_TEFF
bin_edges = np.arange(bin_lo, bin_hi, BIN_WIDTH_TEFF)
bin_ids   = np.digitize(teff_train_raw, bin_edges)

unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
bins_to_drop = set(unique_bins[bin_counts < MIN_BIN_SAMPLES])
keep_mask    = ~np.isin(bin_ids, list(bins_to_drop))

X_train_191    = X_train_191[keep_mask]
y_train        = y_train[keep_mask]
teff_train_raw = teff_train_raw[keep_mask]
st_train       = st_train[keep_mask]

bin_ids                 = np.digitize(teff_train_raw, bin_edges)
unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
max_count               = bin_counts.max()

bin_sigma_features = {}
bin_sigma_logg     = {}
for b in unique_bins:
    mask = bin_ids == b
    bin_sigma_features[b] = np.std(X_train_191[mask], axis=0)
    bin_sigma_logg[b]     = np.std(y_train[mask])

X_train_191_orig = X_train_191.copy()

rng = np.random.default_rng(SEED)
aug_X_list, aug_y_list = [], []

for b, count in zip(unique_bins, bin_counts):
    if count >= max_count:
        continue
    deficit  = max_count - count
    bin_mask = np.where(bin_ids == b)[0]
    src_idx  = rng.choice(bin_mask, size=deficit, replace=True)

    # Noise on all 191 features (including predicted Teff — treated uniformly)
    noise_X    = rng.normal(0.0, bin_sigma_features[b],
                             size=(deficit, X_train_191.shape[1])).astype(np.float32)
    noise_logg = rng.normal(0.0, bin_sigma_logg[b], size=deficit)
    new_logg   = np.clip(y_train[src_idx] + noise_logg, 0.0, None).astype(np.float32)

    aug_X_list.append(X_train_191[src_idx] + noise_X)
    aug_y_list.append(new_logg)

n_augmented  = sum(len(a) for a in aug_y_list)
X_train_aug  = np.concatenate([X_train_191] + aug_X_list)
y_train_aug  = np.concatenate([y_train]     + aug_y_list)

shuffle_idx = rng.permutation(len(y_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]
log(f"Augmented training size: {len(y_train_aug):,}  (+{n_augmented:,} synthetic samples)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
log("Fitting StandardScaler ...")
scaler = StandardScaler()
scaler.fit(X_train_191_orig)

X_train_scaled = scaler.transform(X_train_aug).astype(np.float32)
X_val_scaled   = scaler.transform(X_val_191).astype(np.float32)
X_test_scaled  = scaler.transform(X_test_191).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════
class StellarDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(StellarDataset(X_train_scaled, y_train_aug),
                          batch_size=BATCH_SIZE,   shuffle=True,  num_workers=0)
val_loader   = DataLoader(StellarDataset(X_val_scaled,   y_val),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
test_loader  = DataLoader(StellarDataset(X_test_scaled,  y_test),
                          batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
log(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MODEL  (191 → ResBlock(256) → ResBlock(128) → 64 → 1)
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

class StellarLoggTwoStageNet(nn.Module):
    """
    Residual ANN for log g prediction (Stage 2 of two-stage pipeline).
    Input: 191 features (190 photometric + 1 predicted log10(Teff) from Stage 1).
    191 → ResBlock(256) → ResBlock(128) → Linear(64) → BN → ReLU → Dropout → Linear(1)
    """
    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            ResBlock(input_dim, 256, dropout),
            ResBlock(256, 128, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

model     = StellarLoggTwoStageNet(input_dim=len(FEATURE_COLS_191)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

log(f"Model: StellarLoggTwoStageNet | Params: {sum(p.numel() for p in model.parameters()):,} | Input: {len(FEATURE_COLS_191)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                    patience_counter, train_losses, val_losses, lr_history):
    torch.save({
        "epoch": epoch,
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
# 9. TRAINING LOOP
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
    log(f"Resumed from epoch {ckpt['epoch']}  (best val loss = {best_val_loss:.8f})")
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

hdr = f"{'Epoch':>5}  {'TrLoss(MSE)':>14}  {'VaLoss(MSE)':>14}  {'LR':>10}  Status"
print(hdr);  print("-" * len(hdr))

try:
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0      = time.time()
        tr_loss = train_one_epoch()
        va_loss = evaluate(val_loader)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss);  val_losses.append(va_loss);  lr_history.append(cur_lr)
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
            print(f"{epoch:>5}  {tr_loss:>14.8f}  {va_loss:>14.8f}  {cur_lr:>10.2e}  {status}  [{elapsed:.1f}s]",
                  flush=True)

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, best_model_state, best_val_loss, best_epoch,
                            patience_counter, train_losses, val_losses, lr_history)
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
    log(f"Restored best model from epoch {best_epoch}  (val MSE = {best_val_loss:.8f})")

if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    log("Training complete — checkpoint deleted.")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. TRAINING DIAGNOSTICS PLOT
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving training diagnostics plot ...")
erange = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax = axes[0]
ax.plot(erange, train_losses, label="Train", lw=2)
ax.plot(erange, val_losses,   label="Val",   lw=2)
ax.axvline(best_epoch, color="red", ls="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
ax.set_title("MSE Loss — log g (Two-Stage)"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
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
        all_preds.append(model(X_b.to(DEVICE)).cpu().numpy())
        all_targets.append(y_b.numpy())

logg_pred = np.concatenate(all_preds)
logg_true = np.concatenate(all_targets)
rmse_logg = np.sqrt(mean_squared_error(logg_true, logg_pred))
mae_logg  = mean_absolute_error(logg_true, logg_pred)
r2_logg   = r2_score(logg_true, logg_pred)

print("\n" + "=" * 60)
print("  TEST SET RESULTS — log g Two-Stage ANN")
print("=" * 60)
print(f"  RMSE: {rmse_logg:.4f} dex  |  MAE: {mae_logg:.4f} dex  |  R²: {r2_logg:.5f}")

per_type_metrics = {}
print(f"\n  {'Type':<4} {'N':>8}  {'log g R²':>10}  {'log g RMSE':>12}")
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    if mask.sum() == 0: continue
    lr2   = r2_score(logg_true[mask], logg_pred[mask])
    lrmse = np.sqrt(mean_squared_error(logg_true[mask], logg_pred[mask]))
    per_type_metrics[stype] = {"logg_r2": lr2, "logg_rmse": lrmse}
    print(f"  {stype:<4} {mask.sum():>8,}  {lr2:>10.4f}  {lrmse:>12.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving plots ...")
colors_map = {"F": "#1f77b4", "G": "#2ca02c", "K": "#ff7f0e", "M": "#d62728"}

fig, ax = plt.subplots(figsize=(9, 8))
hb = ax.hexbin(logg_true, logg_pred, gridsize=200, cmap="inferno", mincnt=1, bins="log")
plt.colorbar(hb, ax=ax, label="log10(count)")
lims = [min(logg_true.min(), logg_pred.min()) - 0.1,
        max(logg_true.max(), logg_pred.max()) + 0.1]
ax.plot(lims, lims, "r--", lw=2, alpha=0.8, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("True log g (dex)"); ax.set_ylabel("Predicted log g (dex)")
ax.set_title("log g — Two-Stage ANN (Stage 2)")
ax.text(0.97, 0.03,
        f"RMSE = {rmse_logg:.4f} dex\nMAE = {mae_logg:.4f} dex\nR² = {r2_logg:.5f}",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "one_to_one_plot.png", dpi=200, bbox_inches="tight")
plt.close()

res_logg  = logg_pred - logg_true
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
ax = axes[0]
for stype in ["F", "G", "K", "M"]:
    mask = st_test == stype
    ax.scatter(logg_true[mask], res_logg[mask], alpha=0.05, s=1,
               color=colors_map[stype], label=stype, rasterized=True)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("True log g (dex)"); ax.set_ylabel("Residual (dex)")
ax.set_title("log g Residuals vs True log g"); ax.legend(markerscale=20); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.hist(res_logg, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", ls="--", lw=1.5)
ax.axvline(np.mean(res_logg), color="orange", ls="--", lw=1.5,
           label=f"Mean = {np.mean(res_logg):.4f} dex")
ax.set_xlabel("Residual (dex)"); ax.set_title("log g Residual Distribution")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "residual_plots.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 13. SAVE FINAL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Saving model and metrics ...")
torch.save({
    "model_state_dict": best_model_state,
    "input_dim":        len(FEATURE_COLS_191),
    "feature_cols":     FEATURE_COLS_191,
    "best_epoch":       best_epoch,
    "best_val_loss":    best_val_loss,
    "output":           "logg",
    "stage1_model":     str(STAGE1_MODEL),
    "stage1_variant":   STAGE1_VARIANT,
}, MODELS_DIR / "stellar_logg_twostage_best.pth")

joblib.dump(scaler, MODELS_DIR / "scaler_logg_twostage.pkl")

baselines = {"residual_single_output": {"r2": 0.51924, "rmse": 0.19940},
             "multi_output_homosc":    {"r2": 0.48343, "rmse": 0.20674}}
metrics = {
    "model": "logg_twostage_ann",
    "logg":  {"rmse_dex": round(float(rmse_logg), 6), "mae_dex": round(float(mae_logg), 6),
               "r2_score": round(float(r2_logg), 6)},
    "training": {
        "best_epoch": best_epoch, "total_epochs_run": len(train_losses),
        "n_train_augmented": len(y_train_aug), "n_val": len(y_val), "n_test": len(y_test),
        "n_features": len(FEATURE_COLS_191), "stage1_model": str(STAGE1_MODEL),
        "stage1_variant": STAGE1_VARIANT,
        "augmentation_bin_width_K": BIN_WIDTH_TEFF,
    },
    "per_type": {k: {"logg_r2": round(v["logg_r2"], 6), "logg_rmse": round(v["logg_rmse"], 6)}
                 for k, v in per_type_metrics.items()},
    "baselines": baselines,
}
with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 68)
print("  COMPARISON — log g")
print("=" * 68)
for name, m in baselines.items():
    print(f"  {name:<38}  R²={m['r2']:.5f}  RMSE={m['rmse']:.5f} dex")
flag = "IMPROVED" if r2_logg > 0.51924 else "REGRESSED"
print(f"  {'Two-Stage ANN (this)':<38}  R²={r2_logg:.5f}  RMSE={rmse_logg:.5f} dex"
      f"  ({r2_logg-0.51924:+.5f} R²)  {flag}")
print(f"\n  Artifacts saved to: {MODELS_DIR} and {RESULTS_DIR}")
log("All done.")
