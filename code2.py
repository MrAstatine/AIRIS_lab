import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import math, random, warnings
import optuna
from functools import partial
from scipy.ndimage import gaussian_filter1d
import time

warnings.filterwarnings("ignore")

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_t = (
                self.alpha[targets]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha
            )
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class FastAugmentation:
    """Optimized augmentation with reduced probability and faster operations"""

    def __init__(self, prob=0.6):  # Reduced from 0.8
        self.prob = prob

    def time_warp(self, x, sigma=0.1):  # Reduced sigma
        if random.random() > self.prob:
            return x
        seq_len = x.shape[0]
        # Simplified warping
        warp_strength = random.uniform(-sigma, sigma)
        if abs(warp_strength) < 0.01:
            return x

        indices = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        warp = torch.sin(indices * math.pi / seq_len) * warp_strength * seq_len
        warped_indices = torch.clamp(indices + warp, 0, seq_len - 1)

        # Fast interpolation using gather
        floor_idx = warped_indices.long()
        ceil_idx = torch.clamp(floor_idx + 1, 0, seq_len - 1)
        weight = warped_indices - floor_idx.float()

        return (1 - weight.unsqueeze(1)) * x[floor_idx] + weight.unsqueeze(1) * x[
            ceil_idx
        ]

    def magnitude_warp(self, x, sigma=0.08):  # Reduced sigma
        if random.random() > self.prob:
            return x
        # Simplified magnitude warping
        scale = 1 + torch.normal(0, sigma, (1,), device=x.device)
        return x * scale

    def intelligent_noise(self, x, noise_factor=0.02):  # Reduced noise
        if random.random() > self.prob:
            return x
        noise = torch.normal(0, noise_factor, x.shape, device=x.device)
        return x + noise

    def __call__(self, x):
        # Apply fewer augmentations randomly
        if random.random() < 0.5:
            x = self.time_warp(x)
        if random.random() < 0.5:
            x = self.magnitude_warp(x)
        if random.random() < 0.3:
            x = self.intelligent_noise(x)
        return x


class OptimizedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Smaller learned component
        self.learned_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len] + self.learned_pe[:, :seq_len])


class EfficientMultiScaleCNN(nn.Module):
    """Reduced complexity CNN with fewer scales"""

    def __init__(self, input_channels, d_model):
        super().__init__()
        # Reduced from 3 scales to 2 scales
        self.scale1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, padding=1),  # Reduced channels
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Simplified attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 32, 1),
            nn.GELU(),
            nn.Conv1d(32, 128, 1),
            nn.Sigmoid(),
        )

        self.combine = nn.Sequential(
            nn.Conv1d(128, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        s1, s2 = self.scale1(x), self.scale2(x)
        combined = torch.cat([s1, s2], dim=1)
        attention = self.channel_attention(combined)
        combined = combined * attention
        output = self.combine(combined)
        return output.transpose(1, 2)


class OptimizedTransformer(nn.Module):
    def __init__(
        self,
        input_channels=3,
        d_model=256,  # Reduced default
        nhead=8,
        num_encoder_layers=6,  # Reduced default
        dim_feedforward=1024,  # Reduced default
        dropout=0.1,
        num_classes=3,
        max_seq_len=1000,
    ):
        super().__init__()
        self.feature_extractor = EfficientMultiScaleCNN(input_channels, d_model)
        self.pos_encoder = OptimizedPositionalEncoding(d_model, max_seq_len, dropout)

        # Use flash attention if available
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        # Simplified attention pooling
        self.attention_pool = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim_feedforward // 2, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=0.8)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Attention pooling
        query = self.pool_query.expand(x.size(0), -1, -1)
        pooled, _ = self.attention_pool(query, x, x)
        pooled = pooled.squeeze(1)

        return self.classifier(pooled)


class OptimizedDataset(Dataset):
    def __init__(self, data, labels, normalize=True, augment=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

        if normalize:
            self.data = self._fast_normalize(self.data)

        if augment:
            self.augmenter = FastAugmentation(prob=0.6)

    def _fast_normalize(self, data):
        """Faster normalization using standard methods"""
        # Normalize across the sequence dimension for each channel
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        return (data - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment and random.random() < 0.5:  # Reduced augmentation probability
            x = self.augmenter(x)

        return x, y


def calculate_metrics(y_true, y_pred):
    """Simplified metrics calculation"""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def optimized_hyperparameter_search(data, labels, n_trials=8):  # Reduced trials
    print("🔍 Starting optimized hyperparameter search...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pre-compute data splits to avoid repeated computation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(kf.split(data))

    def objective(trial):
        # Simplified hyperparameter space
        d_model = trial.suggest_categorical("d_model", [256, 288, 320])
        nhead = trial.suggest_categorical("nhead", [8])  # Fixed to reduce search space

        config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": trial.suggest_categorical(
                "num_encoder_layers", [4, 6]
            ),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward", [768, 1024]
            ),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.15]),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 48]
            ),  # Larger batches
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1e-4, 2e-4, 3e-4]
            ),
            "focal_gamma": trial.suggest_categorical("focal_gamma", [2.0, 2.5]),
        }

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_data, val_data = data[train_idx], data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Optimized data loaders
            train_ds = OptimizedDataset(
                train_data, train_labels, normalize=True, augment=True
            )
            val_ds = OptimizedDataset(
                val_data, val_labels, normalize=True, augment=False
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4,  # Increased workers
                pin_memory=True,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=config["batch_size"] * 2,  # Larger validation batch
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            model = OptimizedTransformer(
                input_channels=3,
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_encoder_layers=config["num_encoder_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                num_classes=3,
            ).to(device)

            # Compile model for faster execution (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                model = torch.compile(model)

            class_weights = compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(device)

            criterion = FocalLoss(alpha=class_weights, gamma=config["focal_gamma"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=0.01,
                fused=True,  # Faster optimizer
            )

            best_f1 = 0
            patience = 3  # Reduced patience
            counter = 0
            max_epochs = 15  # Reduced epochs

            for epoch in range(max_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(
                        device, non_blocking=True
                    )
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():  # Mixed precision
                        loss = criterion(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Validation every 2 epochs initially, then every epoch
                if epoch < 5 and epoch % 2 != 0:
                    continue

                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device, non_blocking=True), y.to(
                            device, non_blocking=True
                        )
                        with torch.cuda.amp.autocast():
                            out = model(x)
                        _, pred = out.max(1)
                        preds.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())

                val_f1 = f1_score(targets, preds, average="macro")
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break

            fold_scores.append(best_f1)
            trial.report(best_f1, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n🏆 Best Trial:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"  Value (macro F1): {study.best_trial.value:.4f}")

    return study.best_trial.params, study.best_trial.value, study


# === Load dataset ===
datafile = "final_dataset.npz"
data_npz = np.load(datafile)
print(f"Loaded {datafile}. Keys: {list(data_npz.keys())}")

X = data_npz["data"]
y = data_npz["label"]
print(f"X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    start_time = time.time()
    best_params, best_score, study = optimized_hyperparameter_search(X, y, n_trials=8)
    end_time = time.time()

    print(f"\n⚡ Optimization completed in {end_time - start_time:.2f} seconds")
    print("Best hyperparameters:", best_params)
    print("Best macro F1 score:", best_score)
