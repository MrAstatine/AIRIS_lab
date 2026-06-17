# ------------------------------------------------------------
RUN_ID = "cw_1_6_40"  # change for every new test
# train_clean.py   (one-file, ready to run)
# ------------------------------------------------------------
import numpy as np, random, math, time, argparse, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR


# ============================================================
#  Model building blocks
# ============================================================
class AttentiveMultiScaleCNN(nn.Module):
    def __init__(self, in_ch, d_model):
        super().__init__()

        # three filter‐banks (3,7,11) → 80 channels each
        def block(k):
            return nn.Sequential(
                nn.Conv1d(in_ch, 48, k, padding=k // 2),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Conv1d(48, 64, k, padding=k // 2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 80, k if k > 3 else 3, padding=k // 2),
                nn.BatchNorm1d(80),
                nn.GELU(),
                nn.Conv1d(80, 80, k if k > 3 else 3, padding=k // 2),
                nn.BatchNorm1d(80),
                nn.GELU(),
            )

        self.s1, self.s2, self.s3 = block(3), block(7), block(11)
        self.r1 = nn.Conv1d(in_ch, 80, 1)
        self.r2 = nn.Conv1d(in_ch, 80, 1)
        self.r3 = nn.Conv1d(in_ch, 80, 1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(240, 60, 1),
            nn.GELU(),
            nn.Conv1d(60, 240, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(240, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(self, x):  # B×T×C
        x = x.transpose(1, 2)  # B×C×T
        s1 = self.s1(x) + self.r1(x)
        s2 = self.s2(x) + self.r2(x)
        s3 = self.s3(x) + self.r3(x)
        z = torch.cat([s1, s2, s3], 1)  # B×240×T
        z = z * self.att(z)
        z = self.proj(z).transpose(1, 2)  # B×T×d
        return z


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, p=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(t * div)
        pe[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class F1OptimizedTransformer(nn.Module):
    def __init__(
        self, in_ch=3, d_model=384, nhead=8, nlayers=6, d_ff=1536, n_cls=3, drop=0.2
    ):
        super().__init__()
        self.feat = AttentiveMultiScaleCNN(in_ch, d_model)
        self.pos = PositionalEncoding(d_model, p=drop)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, drop, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc, nlayers)
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=drop
        )
        self.pre = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(drop),
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(d_ff // 2, d_ff // 4),
            nn.GELU(),
            nn.Dropout(drop * 0.25),
            nn.Linear(d_ff // 4, n_cls),
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.8)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):  # B×T×C
        x = self.pos(self.feat(x))
        x = self.enc(x)
        q = self.q.expand(x.size(0), -1, -1)
        x, _ = self.pool(q, x, x)
        x = self.pre(x.squeeze(1))
        return self.head(x)


# ============================================================
#  Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # CE for each element in the batch  →  shape (B,)
        ce = torch.nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )  # keep per-sample

        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce  # shape (B,)
        return focal  # ← NO .mean()


# ============================================================
#  Dataset + normalisation / augmentation
# ============================================================
def robust_norm(a):
    med = np.median(a, axis=0, keepdims=True)
    q1, q3 = np.percentile(a, [25, 75], axis=0, keepdims=True)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    return (a - med) / (1.4826 * iqr)


class TSDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(robust_norm(X)).float()
        self.y = torch.from_numpy(y).long()
        self.aug = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        y = int(self.y[i])
        if self.aug:
            if y == 2:  # rare class
                x = x * (1 + 0.03 * torch.randn_like(x))
            elif y == 1:
                if random.random() < 0.7:
                    x = x * (1 + 0.03 * torch.randn_like(x))
            # y == 0: usually leave untouched
        return x, y


# ============================================================
#  Main training procedure
# ============================================================
def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load and split dataset ---------------------------------------------------
    # Load combined dataset
    combined = np.load("combined_dataset.npz")
    X, y = combined["data"], combined["label"]

    # Calculate split sizes
    total_samples = len(X)
    train_size = int(0.8 * total_samples)  # 80% for training

    # Create indices for random splitting
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Split the data
    Xtr, ytr = X[train_indices], y[train_indices]
    Xva, yva = X[val_indices], y[val_indices]

    print(f"Training samples: {len(Xtr)}, Validation samples: {len(Xva)}")
    print(f"Training class distribution: {np.bincount(ytr)}")
    print(f"Validation class distribution: {np.bincount(yva)}")

    # ------------------------------------------------------------------
    #  Dataset splitting complete, proceed with model training
    # ------------------------------------------------------------------

    from torch.utils.data import RandomSampler

    # build datasets  ──────────────────────────────────────────────
    tr_ds = TSDataset(Xtr, ytr, augment=True)
    va_ds = TSDataset(Xva, yva, augment=False)

    # ---------- NEW balanced-batch sampler ----------
    balanced_sampler = RandomSampler(tr_ds, replacement=True, num_samples=len(tr_ds))

    train_loader = DataLoader(
        tr_ds,
        batch_size=64,
        sampler=balanced_sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        va_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # sanity-check batch balance
    batch_labels = next(iter(train_loader))[1].numpy()
    print("labels in first batch :", np.bincount(batch_labels))

    # ---------- STEP 3: cost-sensitive focal loss -----------------
    # Configure focal loss with class weights based on class distribution

    # α-weights: give bigger penalty to minority classes

    # 2 — Boost focal-loss class weights
    alpha = torch.tensor(
        [
            1.0,  # Noise
            6.0,  # Transit  (stronger weight)
            40.0,
        ],  # EB       (stronger weight)
        device=dev,
    )
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.05)

    # 3 — Downsize / de-regularise the model
    model = F1OptimizedTransformer(
        in_ch=3,
        d_model=256,  # 384 → 256
        nhead=4,  # keep d_model/nhead integer
        nlayers=4,  # 6 → 4
        d_ff=1024,  # scale down feedforward as well
        n_cls=3,
        drop=0.1,  # 0.25 → 0.10
    ).to(dev)

    # 1 — Replace the learning-rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()
    best, bad = 0.0, 0

    for ep in range(1, 41):
        t0 = time.time()
        model.train()
        tot = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                outputs = model(xb)
                loss = criterion(outputs, yb).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            tot += loss.item()

        # ----- validation -----------------------------------------------------
        model.eval()
        preds = []
        targs = []
        with torch.no_grad(), autocast():
            for xb, yb in val_loader:
                out = model(xb.to(dev))
                preds.extend(out.argmax(1).cpu().numpy())
                targs.extend(yb.numpy())
        f1 = f1_score(targs, preds, average="macro")
        print(
            f"Ep {ep:02d}  loss {tot/len(train_loader):.4f}  val F1 {f1:.4f}  "
            f"({time.time()-t0:.1f}s)"
        )

        scheduler.step(f1)

        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            bad += 1
        if bad >= 8:
            print("Early stop")
            break
    print("best val macro-F1:", best)
    with open("runs.txt", "a") as f:
        f.write(f"{RUN_ID}\t{best:.4f}\n")


# ------------------- entry ----------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    main(ap.parse_args())
