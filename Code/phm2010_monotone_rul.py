# phm2010_monotone_rul.py
import os, re, math, random, json, glob
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------
# Helpers for robust path discovery
# --------------------------
def _glob_recursive(base: str, pattern: str):
    # recursive glob
    return sorted(glob.glob(os.path.join(base, pattern), recursive=True))

def discover_wear_file(base: str, cutter_id: int) -> str:
    """
    Try several common patterns for the PHM 2010 milling wear table.
    Searches recursively and accepts CSV or XLS/XLSX.
    Returns absolute path or raises FileNotFoundError.
    """
    cand_patterns = [
        f"**/c{cutter_id}_wear.*",      # c1_wear.csv
        f"**/c_{cutter_id}_wear.*",     # c_1_wear.csv (just in case)
        f"**/*c{cutter_id}*wear*.*",    # anything with c1 and wear
        f"**/*wear*c{cutter_id}*.*",
    ]
    cands = []
    for patt in cand_patterns:
        cands += _glob_recursive(base, patt)

    # prefer csv, then xlsx/xls
    cands_sorted = sorted(cands, key=lambda p: (not p.lower().endswith(".csv"), p.lower()))
    for p in cands_sorted:
        if os.path.isfile(p) and os.path.basename(p).lower().endswith((".csv", ".xlsx", ".xls")):
            return os.path.abspath(p)
    raise FileNotFoundError(f"No wear file found under {base} for cutter {cutter_id}. "
                            f"Tried patterns: {cand_patterns}")

# --------------------------
# CONFIG (robust paths)
# --------------------------
DATA_PATH   = r"E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling"  # parent of c1 etc.
CUTTER_ID   = 1                     # 1, 4, 6 ... (for c1, c4, c6)
WEAR_FILE   = None                  # leave None to auto-discover (recommended). Or set absolute path to wear CSV/XLSX.
SIGNAL_GLOB = "**/c_{}_*.csv".format(CUTTER_ID)  # recursive search for signals, works if files are in root or c1\

SEED = 1337
WINDOW = 256
STRIDE = 64
DOWNSAMPLE = 2
MAX_FILES = None

HORIZON = 150
ENC_HID = 256
LR = 1e-3
BATCH = 64
EPOCHS_PRETEXT = 10
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

L_HI_ORDER = 1.0
L_HAZ_TV   = 1e-3
L_RUL_ASYM = 0.2
EPS_MONO   = 0.0

# --------------------------
# UTILS
# --------------------------
def list_signal_files(base:str, patt:str) -> List[str]:
    # recursive glob so we match files whether they are in root or c{CUTTER_ID}\ subfolder
    files = _glob_recursive(base, patt)
    if MAX_FILES:
        files = files[:MAX_FILES]
    return files

# Forgiving filename parser
_CUT_ID_REGEXES = [
    re.compile(r".*?_([0-9]{3})\.csv$", re.IGNORECASE),  # c_1_001.csv -> 001
    re.compile(r".*?(\d+)\.csv$", re.IGNORECASE),        # fallback: last number group
]
def parse_cut_id(path:str) -> int:
    name = os.path.basename(path)
    for rx in _CUT_ID_REGEXES:
        m = rx.match(name)
        if m:
            return int(m.group(1))
    digits = re.findall(r"(\d+)", name)
    if digits:
        return int(digits[-1])
    raise ValueError(f"Cannot parse cut id from filename: {name}")

def standardize_unit(x:np.ndarray) -> np.ndarray:
    mu, sd = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd

def monotone_isotonic_1d(y:torch.Tensor) -> torch.Tensor:
    """
    Pool-adjacent-violators (PAV) for nonincreasing sequence (HI should go down over time).
    Operates per batch. y shape: [T] or [B,T]
    """
    if y.dim() == 1:
        return _pav_nonincreasing_1d(y)
    out = []
    for i in range(y.shape[0]):
        out.append(_pav_nonincreasing_1d(y[i]))
    return torch.stack(out, 0)

def _pav_nonincreasing_1d(y:torch.Tensor) -> torch.Tensor:
    v = y.detach().cpu().numpy().astype(np.float64)
    x = -v
    n = len(x)
    g = x.copy()
    w = np.ones(n)
    i = 0
    while i < len(g) - 1:
        if g[i] > g[i+1]:  # violation for nondecreasing
            j = i
            while j >= 0 and g[j] > g[j+1]:
                new_w = w[j] + w[j+1]
                new_g = (w[j]*g[j] + w[j+1]*g[j+1]) / new_w
                g[j] = new_g; w[j] = new_w
                g = np.delete(g, j+1); w = np.delete(w, j+1)
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    out = np.empty(n)
    idx = 0
    for gi, wi in zip(g, w):
        out[idx:idx+int(wi)] = gi
        idx += int(wi)
    res = torch.tensor(-out, dtype=y.dtype, device=y.device)
    return res

# --------------------------
# DATA LOADING
# --------------------------
def load_wear_table(base:str, wear_file:str|None, cutter_id:int) -> pd.DataFrame:
    # Determine actual wear path
    if wear_file is None:
        wear_path = discover_wear_file(base, cutter_id)
    else:
        wear_path = wear_file if os.path.isabs(wear_file) else os.path.join(base, wear_file)
        if not os.path.exists(wear_path):
            wear_path = discover_wear_file(base, cutter_id)

    print(f"[INFO] Using wear table: {wear_path}")

    # Load CSV or Excel
    lower = wear_path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(wear_path, sep=None, engine="python", encoding_errors="ignore")
    elif lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(wear_path)
    else:
        raise ValueError(f"Unsupported wear file extension: {wear_path}")

    # Normalize and compute wear
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "cut" not in df.columns:
        alt_cut = [c for c in df.columns if c.startswith("cut")]
        if alt_cut:
            df.rename(columns={alt_cut[0]:"cut"}, inplace=True)
        else:
            raise ValueError(f"'cut' column not found in wear file. Columns={df.columns}")

    flute_cols = [c for c in df.columns if c.startswith("flute")]
    if not flute_cols:
        if "wear" in df.columns:
            df["wear"] = df["wear"].astype(float)
        else:
            alt = [c for c in df.columns if "wear" in c or "vb" in c]
            if not alt:
                raise ValueError(f"No flute_* / wear columns found. Columns={df.columns}")
            df["wear"] = df[alt].max(axis=1)
    else:
        df["wear"] = df[flute_cols].max(axis=1)

    df["cut"] = df["cut"].astype(int)
    return df[["cut","wear"]].sort_values("cut").reset_index(drop=True)

def load_and_window_file(path:str, window:int, stride:int, downsample:int) -> Tuple[np.ndarray, List[int]]:
    raw = pd.read_csv(path)
    X = raw.values.astype(np.float32)
    if downsample and downsample > 1:
        X = X[::downsample]
    # if the recording is too short, skip windowing
    if X.shape[0] < window:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), []
    X = standardize_unit(X)
    starts = list(range(0, X.shape[0]-window+1, stride))
    if len(starts) == 0:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), []
    windows = np.stack([X[s:s+window] for s in starts], axis=0)
    return windows, starts

def build_dataset(base:str, wear_file:str|None, pattern:str, window:int, stride:int, downsample:int, cutter_id:int):
    # Sanity prints
    print("[SANITY] DATA_PATH =", base)
    print("[SANITY] SIGNAL_GLOB =", pattern)

    wear = load_wear_table(base, wear_file, cutter_id)  # cut → wear value (robust)
    files = list_signal_files(base, pattern)
    print(f"[INFO] Found {len(files)} signal files for pattern={pattern}")
    if not files:
        raise FileNotFoundError(f"No signal files matched. Base={base} Pattern={pattern}")

    X_all, y_all, cut_ids, starts_all = [], [], [], []

    for f in tqdm(files, desc="Windowing"):
        try:
            cut_id = parse_cut_id(f)
        except Exception as e:
            print(f"[WARN] Skipping file (cannot parse cut id): {f}  reason={e}")
            continue

        windows, starts = load_and_window_file(f, window, stride, downsample)
        if windows.shape[0] == 0:
            print(f"[WARN] Skipping file (no windows after windowing): {f}")
            continue

        # map wear by cut id
        row = wear[wear['cut'] == cut_id]
        if len(row) == 0:
            print(f"[WARN] No wear row for cut={cut_id} -> skipping file {os.path.basename(f)}")
            continue

        # Create monotone pseudo-RUL inside each run
        T = len(starts)
        rul_series = np.linspace(T, 1, T, dtype=np.float32)

        X_all.append(windows)
        y_all.append(rul_series[:, None])  # [T,1]
        cut_ids += [cut_id]*T
        starts_all += starts

    if not X_all:
        raise RuntimeError("No windows built. Check file pattern and wear mapping.")

    X = np.concatenate(X_all, 0)           # [N, W, C]
    y = np.concatenate(y_all, 0)           # [N, 1]
    y = np.minimum(y, HORIZON).astype(np.float32)
    return X, y, np.array(cut_ids), np.array(starts_all)

class PHMWindows(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # [N, W, C]
        self.y = torch.from_numpy(y)  # [N, 1]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        # return sequence [C, W] for conv1d
        return self.X[i].transpose(0,1), self.y[i]

# --------------------------
# SELF-SUPERVISED PRETRAIN (TS-contrastive)
# --------------------------
class TCNEncoder(nn.Module):
    def __init__(self, C_in:int, hid:int=ENC_HID):
        super().__init__()
        ch = [C_in, 64, 128, hid]
        layers = []
        dil = 1
        for i in range(len(ch)-1):
            layers += [
                nn.Conv1d(ch[i], ch[i+1], kernel_size=3, padding=dil, dilation=dil),
                nn.ReLU(),
                nn.GroupNorm(8, ch[i+1])
            ]
            dil *= 2
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):  # x: [B,C,W]
        h = self.net(x)    # [B,H,W]
        g = self.pool(h).squeeze(-1)  # [B,H]
        return g

def augment_time(x:torch.Tensor) -> torch.Tensor:
    # simple jitters & masking
    B,C,W = x.shape
    x = x.clone()
    # random channel dropout
    if random.random()<0.5 and C>0:
        ch = random.randrange(C)
        x[:,ch,:] = 0
    # random time masking
    if random.random()<0.5 and W>8:
        w = random.randint(max(1,W//16), max(2,W//8))
        s = random.randint(0, max(0, W-w))
        x[:,:,s:s+w] = 0
    return x

def nt_xent(z1, z2, temp=0.2):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

# --------------------------
# MONOTONE HI + SURVIVAL HEAD
# --------------------------
class HIHead(nn.Module):
    def __init__(self, d_in:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, g):   # [B,d]
        hi = self.mlp(g)   # [B,1]
        return hi

class SurvivalHead(nn.Module):
    """
    Predict hazards h_tau for tau=1..H with non-decreasing constraint:
    param -> softplus -> cumulative sum -> sigmoid => monotone hazards
    """
    def __init__(self, d_in:int, H:int):
        super().__init__()
        self.fc = nn.Linear(d_in, H)
    def forward(self, g):  # [B,d]
        raw = self.fc(g)                  # [B,H]
        inc = F.softplus(raw)             # >=0
        cum = torch.cumsum(inc, dim=-1)   # nondecreasing
        hazards = torch.sigmoid(cum/10.0) # [B,H]
        return hazards

def survival_to_expected_rul(h):
    # h: [B,H], S(tau) = prod_{k<=tau} (1 - h_k)
    S = torch.cumprod(1.0 - h + 1e-8, dim=-1)  # [B,H]
    ERUL = torch.sum(S, dim=-1, keepdim=True)  # [B,1]
    return S, ERUL

# --------------------------
# TRAIN / EVAL
# --------------------------
def train_ssl(encoder, loader, epochs=EPOCHS_PRETEXT):
    enc = encoder.to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)
    enc.train()
    for ep in range(1, epochs+1):
        losses = []
        for x,_ in tqdm(loader, desc=f"SSL epoch {ep}/{epochs}"):
            x = x.to(DEVICE) # [B,C,W]
            x1 = augment_time(x); x2 = augment_time(x)
            z1 = enc(x1); z2 = enc(x2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"SSL ep{ep}: {np.mean(losses):.4f}")
    return enc

def train_supervised(encoder, hi_head, surv_head, train_loader, val_loader, epochs=EPOCHS):
    encoder.to(DEVICE); hi_head.to(DEVICE); surv_head.to(DEVICE)
    params = list(encoder.parameters()) + list(hi_head.parameters()) + list(surv_head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    best = {"val": 1e9, "state": None}
    for ep in range(1, epochs+1):
        encoder.train(); hi_head.train(); surv_head.train()
        tr_losses = []
        for x, y_rul in tqdm(train_loader, desc=f"Train epoch {ep}/{epochs}"):
            x = x.to(DEVICE); y_rul = y_rul.to(DEVICE).squeeze(-1)  # [B]
            g = encoder(x)                         # [B,d]
            hi = hi_head(g).squeeze(-1)           # [B]
            hazards = surv_head(g)                 # [B,H]
            S, ERUL = survival_to_expected_rul(hazards)  # ERUL: [B,1]
            ERUL = ERUL.squeeze(-1)               # [B]

            # 1) Robust L1 + asymmetric penalty (underestimation late is worse)
            l1 = torch.abs(ERUL - y_rul)
            asym = torch.where(ERUL < y_rul, 1.5*l1, 1.0*l1).mean()

            # 2) HI order loss: encourage HI_t >= HI_{t+1}
            idx = torch.argsort(y_rul, descending=True)  # early->late
            hi_sorted = hi[idx]
            d = hi_sorted[:-1] - hi_sorted[1:] + EPS_MONO
            order_loss = F.relu(d).mean()

            # 3) Hazards total variation smoothing
            tv = torch.mean(torch.abs(hazards[:,1:] - hazards[:,:-1]))

            # 4) RUL monotonicity across adjacent windows within batch
            erul_sorted = ERUL[idx]
            d_erul = erul_sorted[:-1] - erul_sorted[1:] + EPS_MONO
            rul_mono = F.relu(-d_erul).mean()  # penalize increases

            loss = asym + L_HI_ORDER*order_loss + L_HAZ_TV*tv + L_RUL_ASYM*rul_mono

            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(loss.item())

        # Validation
        encoder.eval(); hi_head.eval(); surv_head.eval()
        with torch.no_grad():
            val_losses, mons = [], []
            for x, y_rul in val_loader:
                x = x.to(DEVICE); y_rul = y_rul.to(DEVICE).squeeze(-1)
                g = encoder(x); hi = hi_head(g).squeeze(-1); hazards = surv_head(g)
                _, ERUL = survival_to_expected_rul(hazards)
                ERUL = ERUL.squeeze(-1)
                val_losses.append(F.l1_loss(ERUL, y_rul).item())
                # batch monotonicity index: fraction of nonincreasing ERUL
                idx = torch.argsort(y_rul, descending=True)
                erul_sorted = ERUL[idx]
                ok = (erul_sorted[:-1] >= erul_sorted[1:] - 1e-6).float().mean().item()
                mons.append(ok)
            v = np.mean(val_losses); m = np.mean(mons)
        print(f"Epoch {ep}: train {np.mean(tr_losses):.4f}  val_L1 {v:.4f}  mono_idx {m:.4f}")

        if v < best["val"]:
            best["val"] = v
            best["state"] = {
                "encoder": encoder.state_dict(),
                "hi": hi_head.state_dict(),
                "surv": surv_head.state_dict(),
            }

    return best

def evaluate(encoder, hi_head, surv_head, loader, apply_isotonic_hi=True):
    encoder.eval(); hi_head.eval(); surv_head.eval()
    all_ERUL, all_y, all_HI = [], [], []
    with torch.no_grad():
        for x, y_rul in loader:
            x = x.to(DEVICE); g = encoder(x)
            hi = hi_head(g).squeeze(-1)  # [B]
            if apply_isotonic_hi:
                hi = monotone_isotonic_1d(hi)    # perfect monotone within batch order
            hazards = surv_head(g)
            _, ERUL = survival_to_expected_rul(hazards)
            all_ERUL.append(ERUL.cpu()); all_y.append(y_rul); all_HI.append(hi.cpu().unsqueeze(-1))
    ERUL = torch.cat(all_ERUL,0).squeeze(-1).numpy()
    Y    = torch.cat(all_y,0).squeeze(-1).numpy()
    HI   = torch.cat(all_HI,0).squeeze(-1).numpy()
    mae = np.mean(np.abs(ERUL - Y))
    # monotonicity index (sequence-level approximation via sorting by true time)
    idx = np.argsort(-Y)  # early->late
    mono = np.mean(ERUL[idx][:-1] >= ERUL[idx][1:] - 1e-6)
    return {"MAE": float(mae), "mono_idx": float(mono)}, ERUL, Y, HI

# --------------------------
# CONFORMAL INTERVALS (split conformal on residuals)
# --------------------------
def conformal_calibration(preds_tr, y_tr, alpha=0.1):
    resid = np.abs(preds_tr - y_tr)
    q = np.quantile(resid, 1 - alpha)
    return float(q)

def conformal_interval(pred, q):
    return (float(pred - q), float(pred + q))

# --------------------------
# MAIN
# --------------------------
def main():
    print("Building windows…")
    # Build dataset (now passes CUTTER_ID)
    X, y, cut_ids, starts = build_dataset(DATA_PATH, WEAR_FILE, SIGNAL_GLOB,
                                          WINDOW, STRIDE, DOWNSAMPLE, CUTTER_ID)
    N, W, C = X.shape
    print(f"Windows: {N}  Window:{W}  Channels:{C}")

    ds = PHMWindows(X, y)
    n_cal = int(0.15*len(ds))
    n_val = int(0.15*len(ds))
    n_train = len(ds) - n_cal - n_val
    ds_train, ds_cal, ds_val = random_split(ds, [n_train, n_cal, n_val], generator=torch.Generator().manual_seed(SEED))

    # SSL pretrain
    ssl_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    sample_x, _ = ds[0]
    C_in = sample_x.shape[0]
    encoder = TCNEncoder(C_in, ENC_HID)
    print("Self-supervised pretraining…")
    train_ssl(encoder, ssl_loader, epochs=EPOCHS_PRETEXT)

    # Supervised heads
    hi_head = HIHead(ENC_HID)
    surv_head = SurvivalHead(ENC_HID, HORIZON)

    train_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH, shuffle=False)

    print("Supervised training…")
    best = train_supervised(encoder, hi_head, surv_head, train_loader, val_loader, epochs=EPOCHS)

    # Load best
    if best["state"] is not None:
        encoder.load_state_dict(best["state"]["encoder"])
        hi_head.load_state_dict(best["state"]["hi"])
        surv_head.load_state_dict(best["state"]["surv"])

    # Calibration split for conformal
    cal_loader = DataLoader(ds_cal, batch_size=BATCH, shuffle=False)
    metrics_cal, preds_cal, y_cal, _ = evaluate(encoder, hi_head, surv_head, cal_loader, apply_isotonic_hi=True)
    q = conformal_calibration(preds_cal, y_cal, alpha=0.1)
    print(f"Calibration MAE={metrics_cal['MAE']:.4f}, mono_idx={metrics_cal['mono_idx']:.4f}, q={q:.3f}")

    # Final validation
    metrics_val, preds_val, y_val, hi_val = evaluate(encoder, hi_head, surv_head, val_loader, apply_isotonic_hi=True)
    print(f"VAL   MAE={metrics_val['MAE']:.4f}, mono_idx={metrics_val['mono_idx']:.4f}")

    # Example interval
    if len(preds_val)>0:
        p0, y0 = preds_val[0], y_val[0]
        L0,U0 = conformal_interval(p0, q)
        print(f"Example: pred={p0:.1f}, true={y0:.1f}, 90% PI=({L0:.1f}, {U0:.1f})")

    # Save artifacts
    out = {
        "config": dict(WINDOW=WINDOW, STRIDE=STRIDE, HORIZON=HORIZON, ENC_HID=ENC_HID),
        "metrics_val": metrics_val,
        "conformal_q": q
    }
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "hi_head": hi_head.state_dict(),
                "surv_head": surv_head.state_dict()}, "artifacts/phm2010_monotone.pt")
    with open("artifacts/summary.json","w") as f:
        json.dump(out, f, indent=2)
    print("Saved to artifacts/")

if __name__ == "__main__":
    main()
