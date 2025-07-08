import os, random, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

torch.backends.cudnn.benchmark = True  # speed

# ---------- 1. Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- 2. Paths ----------
DATA_PATH = "combined_dataset.npz"  # CHANGE if needed
OUT_DIR = "Again_output"  # CHANGE if needed
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 3. Load data ----------
npz = np.load(DATA_PATH)
X = npz["data"]  # shape (N, 1000, C)
y = npz["label"]  # shape (N,)
assert len(X) == len(y)  # sanity
C = X.shape[2]
print("Dataset:", X.shape, "classes:", np.bincount(y))


# ---------- 4. Robust normalisation ----------
def robust_norm(arr):
    arr = arr.astype(np.float32)
    out = arr.copy()
    for c in range(arr.shape[2]):
        flat = arr[:, :, c].reshape(-1)
        q25, q75 = np.percentile(flat, [25, 75])
        iqr = q75 - q25 if q75 > q25 else flat.std()
        med = np.median(flat)
        out[:, :, c] = (arr[:, :, c] - med) / (1.4826 * iqr + 1e-6)
    return out


X = robust_norm(X)


# ---------- 5. Basic augmentation ----------
def augment(batch):  # batch: (B, 1000, C)
    if random.random() < 0.50:
        # time-flip
        batch = torch.flip(batch, dims=[1])
    if random.random() < 0.50:
        # small Gaussian noise
        noise = torch.randn_like(batch) * 0.02
        batch = batch + noise
    return batch


# ---------- 6. Dataset & balanced sampler ----------
class CurveDS(Dataset):
    def __init__(self, data, labels, train=True):
        self.x = torch.from_numpy(data)
        self.y = torch.from_numpy(labels).long()
        self.train = train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.train:
            x = augment(x)
        return x, self.y[idx]


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch = batch_size
        self.cls = np.unique(labels)
        self.per = batch_size // len(self.cls)
        self.idxs = {c: np.where(self.labels == c)[0] for c in self.cls}
        self.steps = len(labels) // batch_size

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            for c in self.cls:
                batch.extend(np.random.choice(self.idxs[c], self.per, replace=True))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps


# ---------- 7. Simple CNN + GAP model (fast to train) ----------
class SmallCNN(nn.Module):
    def __init__(self, channels=C, nclass=3):
        super()._init_()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, nclass)
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.transpose(1, 2)  # (B,C,T)
        x = self.net(x)
        x = x.mean(dim=2)  # GAP
        return self.head(x)


# ---------- 8. Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super()._init_()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, tgt):
        ce = nn.functional.cross_entropy(
            logits, tgt, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------- 9. Train / Val split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

bs = 24
train_ds = CurveDS(X_train, y_train, train=True)
val_ds = CurveDS(X_val, y_val, train=False)
train_ld = DataLoader(
    train_ds, batch_sampler=BalancedBatchSampler(y_train, bs), num_workers=2
)
val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

# ----------10. Model, loss, optimiser ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN().to(device)
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weights = torch.tensor(weights, dtype=torch.float32, device=device)
criterion = FocalLoss(weights, gamma=2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, steps_per_epoch=len(train_ld), epochs=300, pct_start=0.3
)

# ----------11. Training loop ----------
best_f1, patience_left = 0.0, 30  # Increased patience
for epoch in range(300):
    # ---- train ----
    model.train()
    tot = 0
    for xb, yb in train_ld:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item()
    # ---- validate ----
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for xb, yb in val_ld:
            xb = xb.to(device)
            out = model(xb).softmax(1).cpu().numpy()
            preds.extend(out)
            targs.extend(yb.numpy())
    preds = np.array(preds)
    targs = np.array(targs)

    # per-class threshold tuning on validation set
    best_thr = np.zeros(preds.shape[1])
    preds_bin = np.zeros_like(preds)
    for c in range(preds.shape[1]):
        thr, best = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.05):
            pr = preds.argmax(1) if c == 0 else preds.argmax(1)  # placeholder
        # simple argmax (multi-class) is fine here:
    y_pred = preds.argmax(1)
    f1 = f1_score(targs, y_pred, average="macro")
    print(f"Epoch {epoch+1:02d}  loss {tot/len(train_ld):.4f}  val-F1 {f1:.4f}")

    if f1 > best_f1:
        best_f1, patience_left = f1, 30  # Reset patience
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
    else:
        patience_left -= 1
        if patience_left == 0:
            break

print("\nBest macro-F1 =", round(best_f1, 4))

# ----------12. Confusion matrix ----------
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in val_ld:
        preds.extend(model(xb.to(device)).softmax(1).cpu().numpy())
y_pred = np.array(preds).argmax(1)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Noise", "Planet", "EB"])
disp.plot(cmap="Blues")
plt.title("Validation confusion matrix")
plt.savefig(os.path.join(OUT_DIR, "confusion.png"))
plt.close()
print("Confusion matrix saved to", OUT_DIR)
