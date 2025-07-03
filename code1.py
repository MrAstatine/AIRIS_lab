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
    # classification_report,
    confusion_matrix,
)
import math, random, warnings
import optuna
from functools import partial

# from scipy import signal
from scipy.ndimage import gaussian_filter1d  # <-- Fix import

warnings.filterwarnings("ignore")


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


class F1OptimizedAugmentation:
    def __init__(self, prob=0.8):
        self.prob = prob

    def time_warp(self, x, sigma=0.15):
        if random.random() > self.prob:
            return x
        seq_len = x.shape[0]
        tt = torch.arange(seq_len).float()
        warp = torch.normal(0, sigma, (seq_len,)).cumsum(0)
        warp = (warp - warp.mean()) / warp.std() * sigma
        warped_tt = torch.clamp(tt + warp, 0, seq_len - 1)
        indices = warped_tt.long()
        weights = warped_tt - indices.float()
        indices_next = torch.clamp(indices + 1, 0, seq_len - 1)
        warped_x = x.clone()
        for c in range(x.shape[1]):
            warped_x[:, c] = (1 - weights) * x[indices, c] + weights * x[
                indices_next, c
            ]
        return warped_x

    def magnitude_warp(self, x, sigma=0.12):
        if random.random() > self.prob:
            return x
        seq_len = x.shape[0]
        knots = torch.randint(3, 6, (1,)).item()
        warped_x = x.clone()
        for c in range(x.shape[1]):
            knot_pos = torch.linspace(0, seq_len - 1, knots)
            knot_val = torch.normal(1.0, sigma, (knots,))
            warp_curve = torch.zeros(seq_len)
            for i in range(seq_len):
                li = torch.searchsorted(knot_pos, float(i), right=False) - 1
                li = torch.clamp(li, 0, knots - 2)
                ri = li + 1
                t = (i - knot_pos[li]) / (knot_pos[ri] - knot_pos[li])
                warp_curve[i] = (1 - t) * knot_val[li] + t * knot_val[ri]
            warp_curve = torch.from_numpy(
                gaussian_filter1d(warp_curve.numpy(), 1.5)  # <-- Fix usage
            ).float()
            warped_x[:, c] *= warp_curve
        return warped_x

    def intelligent_noise(self, x, noise_factor=0.03):
        if random.random() > self.prob:
            return x
        signal_std = x.std(dim=0, keepdim=True)
        noise = torch.normal(0, noise_factor * signal_std, x.shape)
        return x + noise

    def selective_cutout(self, x, max_holes=2, max_length=15):
        if random.random() > self.prob:
            return x
        seq_len, augmented = x.shape[0], x.clone()
        variance = torch.var(x, dim=1)
        threshold = torch.quantile(variance, 0.7)
        for _ in range(random.randint(1, max_holes)):
            length = random.randint(3, max_length)
            for _ in range(10):
                start = random.randint(0, max(1, seq_len - length))
                if variance[start : start + length].mean() < threshold:
                    break
            fill = (
                (
                    x[start - 2 : start].mean(0)
                    + x[start + length : start + length + 2].mean(0)
                )
                / 2
                if start > 2 and start + length < seq_len - 2
                else x.mean(0)
            )
            augmented[start : start + length] = fill
        return augmented

    def __call__(self, x):
        return self.selective_cutout(
            self.intelligent_noise(self.magnitude_warp(self.time_warp(x)))
        )


class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.15):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.learned_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return self.dropout(
            x + self.pe[:, : x.size(1)] + self.learned_pe[:, : x.size(1)]
        )


class AttentiveMultiScaleCNN(nn.Module):
    def __init__(self, input_channels, d_model):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv1d(input_channels, 48, 3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 80, 3, padding=1),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(input_channels, 48, 7, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 80, 5, padding=2),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.scale3 = nn.Sequential(
            nn.Conv1d(input_channels, 48, 11, padding=5),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 80, 7, padding=3),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(240, 60, 1),
            nn.GELU(),
            nn.Conv1d(60, 240, 1),
            nn.Sigmoid(),
        )
        self.combine = nn.Sequential(
            nn.Conv1d(240, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        s1, s2, s3 = self.scale1(x), self.scale2(x), self.scale3(x)
        combined = torch.cat([s1, s2, s3], dim=1)
        attention = self.channel_attention(combined)
        combined = combined * attention
        output = self.combine(combined)
        return output.transpose(1, 2)


class F1OptimizedTransformer(nn.Module):
    def __init__(
        self,
        input_channels=3,
        d_model=320,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=1280,
        dropout=0.15,
        num_classes=3,
        max_seq_len=1000,
    ):
        super().__init__()
        self.feature_extractor = AttentiveMultiScaleCNN(input_channels, d_model)
        self.pos_encoder = EnhancedPositionalEncoding(d_model, max_seq_len, dropout)
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
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention_pool = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(dim_feedforward // 4, num_classes),
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
        query = self.pool_query.expand(x.size(0), -1, -1)
        pooled, _ = self.attention_pool(query, x, x)
        pooled = pooled.squeeze(1)
        pooled = pooled + self.pre_classifier(pooled)
        return self.classifier(pooled)


class F1FocusedDataset(Dataset):
    def __init__(self, data, labels, normalize=True, augment=False, class_weights=None):
        self.original_data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.class_weights = class_weights
        self.data = (
            self._robust_normalize(self.original_data)
            if normalize
            else self.original_data.clone()
        )
        if augment:
            self.augmenter = F1OptimizedAugmentation(prob=0.85)

    def _robust_normalize(self, data):
        normed = data.clone()
        for c in range(data.shape[2]):
            channel_data = data[:, :, c]
            q05, q25, q50, q75, q95 = torch.quantile(
                channel_data.flatten(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
            )
            iqr = q75 - q25
            clipped = torch.clamp(channel_data, q05, q95)
            if iqr > 1e-6:
                normed[:, :, c] = (clipped - q50) / (1.4826 * iqr)
            else:
                mean, std = clipped.mean(), clipped.std()
                normed[:, :, c] = (clipped - mean) / (std + 1e-8)
        return normed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        y = self.labels[idx]
        if self.augment:
            aug_prob = (
                0.9
                if self.class_weights is not None and self.class_weights[y] > 1.5
                else 0.7
            )
            if random.random() < aug_prob:
                x = self.augmenter(x)
        return x, y


def calculate_comprehensive_metrics(y_true, y_pred, class_names=None):
    class_names = class_names or ["Noise", "Planetary Transit", "Eclipsing Binary"]
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "per_class_f1": f1_score(y_true, y_pred, average=None),
        "per_class_precision": precision_score(y_true, y_pred, average=None),
        "per_class_recall": recall_score(y_true, y_pred, average=None),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return metrics


def hyperparameter_search(data, labels, n_trials=10):
    print("🔍 Starting hyperparameter search...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def objective(trial, data, labels):
        config = {
            "d_model": trial.suggest_int("d_model", 256, 384, step=32),
            "nhead": trial.suggest_int("nhead", 4, 12, step=2),
            "num_encoder_layers": trial.suggest_int(
                "num_encoder_layers", 4, 10, step=2
            ),
            "dim_feedforward": trial.suggest_int(
                "dim_feedforward", 768, 1536, step=256
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.2),
            "batch_size": trial.suggest_int("batch_size", 16, 32, step=8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
            "focal_gamma": trial.suggest_float("focal_gamma", 2.0, 3.0),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.05, 0.15),
            "augment_prob": trial.suggest_float("augment_prob", 0.7, 0.9),
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"\nEvaluating Fold {fold+1}/3")
            train_data, val_data = data[train_idx], data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            train_ds = F1FocusedDataset(
                train_data, train_labels, normalize=True, augment=True
            )
            val_ds = F1FocusedDataset(
                val_data, val_labels, normalize=True, augment=False
            )
            train_loader = DataLoader(
                train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2
            )

            model = F1OptimizedTransformer(
                input_channels=3,
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_encoder_layers=config["num_encoder_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                num_classes=3,
            ).to(device)

            class_weights = compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(device)

            criterion = FocalLoss(
                alpha=class_weights,
                gamma=config["focal_gamma"],
                label_smoothing=config["label_smoothing"],
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["learning_rate"], weight_decay=0.01
            )

            best_f1 = 0
            patience = 5
            counter = 0

            for _ in range(30):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
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
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(partial(objective, data=data, labels=labels), n_trials=n_trials)

    print("\n🏆 Best Trial:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"  Value (macro F1): {study.best_trial.value:.4f}")
    return study.best_trial.params, study.best_trial.value, study


# === Load dataset ===
datafile = "final_dataset.npz"
data_npz = np.load(datafile)
print(f"Loaded {datafile}. Keys: {list(data_npz.keys())}")

# Adjust these keys if your .npz file uses different names
X = data_npz["data"]
y = data_npz["label"]
print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    # Start processing the dataset with hyperparameter search
    best_params, best_score, study = hyperparameter_search(X, y, n_trials=10)
    print("Best hyperparameters:", best_params)
    print("Best macro F1 score:", best_score)
