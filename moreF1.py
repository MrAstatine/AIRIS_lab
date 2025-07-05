import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import gaussian_filter1d
import numpy as np
import random

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume data and label are loaded
# data: shape (2591, 1000, 3), label: shape (2591,)
X = data
y = label

# Verify shapes and class distribution
print("Data shape:", X.shape)
print("Label shape:", y.shape)
print("Class counts:", np.bincount(y))


# F1OptimizedAugmentation
class F1OptimizedAugmentation:
    def _init_(self, prob=0.95):
        self.prob = prob

    def time_warp(self, x, is_minority=False):
        sigma = 0.25 if is_minority else 0.15
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

    def magnitude_warp(self, x, is_minority=False):
        sigma = 0.2 if is_minority else 0.12
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
                gaussian_filter1d(warp_curve.numpy(), 1.5)
            ).float()
            warped_x[:, c] *= warp_curve
        return warped_x

    def intelligent_noise(self, x, is_minority=False):
        noise_factor = 0.05 if is_minority else 0.03
        if random.random() > self.prob:
            return x
        signal_std = x.std(dim=0, keepdim=True)
        noise = torch.normal(0.0, noise_factor * signal_std.expand_as(x))
        return x + noise

    def selective_cutout(self, x, is_minority=False):
        max_holes = 4 if is_minority else 2
        max_length = 20 if is_minority else 15
        if random.random() > self.prob:
            return x
        seq_len, augmented = x.shape[0], x.clone()
        variance = torch.var(x[:, 0], dim=0)  # Focus on flux channel
        threshold = torch.quantile(variance, 0.7)
        for _ in range(random.randint(1, max_holes)):
            length = random.randint(3, max_length)
            for _ in range(10):
                start = random.randint(0, max(1, seq_len - length))
                if variance[start : start + length].mean() < threshold:
                    break
            fill = (
                (
                    x[start - 2 : start, :].mean(0)
                    + x[start + length : start + length + 2, :].mean(0)
                )
                / 2
                if start > 2 and start + length < seq_len - 2
                else x.mean(0)
            )
            augmented[start : start + length] = fill
        return augmented

    def _call_(self, x, is_minority=False):
        return self.selective_cutout(
            self.intelligent_noise(
                self.magnitude_warp(self.time_warp(x, is_minority), is_minority),
                is_minority,
            ),
            is_minority,
        )


# F1FocusedDataset
class F1FocusedDataset(Dataset):
    def _init_(self, data, labels, normalize=True, augment=False, class_weights=None):
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
            self.augmenter = F1OptimizedAugmentation(prob=0.95)
        self.minority_class = 2

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

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        x = self.data[idx].clone()
        y = self.labels[idx]
        if self.augment:
            is_minority = y == self.minority_class
            aug_prob = 0.98 if is_minority else 0.7
            if random.random() < aug_prob:
                x = self.augmenter(x, is_minority=is_minority)
        return x, y


# AttentiveMultiScaleCNN
class AttentiveMultiScaleCNN(nn.Module):
    def _init_(self, input_channels, d_model):
        super()._init_()
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
            nn.Conv1d(80, 80, 3, padding=1),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.residual1 = nn.Conv1d(input_channels, 80, 1)
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
            nn.Conv1d(80, 80, 5, padding=2),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.residual2 = nn.Conv1d(input_channels, 80, 1)
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
            nn.Conv1d(80, 80, 7, padding=3),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.residual3 = nn.Conv1d(input_channels, 80, 1)
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
            nn.Dropout(0.25),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        s1 = self.scale1(x) + self.residual1(x)
        s2 = self.scale2(x) + self.residual2(x)
        s3 = self.scale3(x) + self.residual3(x)
        combined = torch.cat([s1, s2, s3], dim=1)
        attention = self.channel_attention(combined)
        combined = combined * attention
        output = self.combine(combined)
        return output.transpose(1, 2)


# EnhancedPositionalEncoding
class EnhancedPositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=1000, dropout=0.1):
        super()._init_()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# F1OptimizedTransformer
class F1OptimizedTransformer(nn.Module):
    def _init_(
        self,
        input_channels=3,
        d_model=384,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1536,
        dropout=0.2,
        num_classes=3,
        max_seq_len=1000,
    ):
        super()._init_()
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
        batch_size = x.size(0)
        query = self.pool_query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention_pool(query, x, x)
        x = self.pre_classifier(attn_output.squeeze(1))
        x = self.classifier(x)
        return x


# FocalLoss
class FocalLoss(nn.Module):
    def _init_(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super()._init_()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# WeightedRandomSampler
class_counts = np.bincount(y)
sample_weights = 1.0 / class_counts[y]
sample_weights[y == 2] *= 2.0
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(y), replacement=True
)

# DataLoaders
train_dataset = F1FocusedDataset(X, y, normalize=True, augment=True)
val_dataset = F1FocusedDataset(X, y, normalize=True, augment=False)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Initialize model
model = F1OptimizedTransformer(
    input_channels=3,
    d_model=384,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=1536,
    dropout=0.2,
    num_classes=3,
).to(device)

# Class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights[2] *= 2.0
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class weights:", class_weights)

# Loss and optimizer
criterion = FocalLoss(alpha=class_weights, gamma=3.0, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.02)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)

# Training loop
best_f1 = 0
patience = 10
counter = 0
train_losses, val_f1s = [], []
for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss_weights = torch.ones_like(y, dtype=torch.float, device=device)
        loss_weights[y == 2] = 2.0
        loss = (loss * loss_weights).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

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
    val_f1s.append(val_f1)
    scheduler.step(val_f1)
    print(f"Epoch {epoch+1}/50, Loss: {train_losses[-1]:.4f}, Val F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping")
        break


# Evaluate
def calculate_comprehensive_metrics(targets, preds):
    macro_f1 = f1_score(targets, preds, average="macro")
    per_class_f1 = f1_score(targets, preds, average=None)
    per_class_precision = precision_score(targets, preds, average=None)
    per_class_recall = recall_score(targets, preds, average=None)
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, preds)
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion_matrix,
    }


metrics = calculate_comprehensive_metrics(targets, preds)
class_names = ["Noise", "Planetary Transit", "Eclipsing Binary"]
print("Final Macro F1:", metrics["macro_f1"])
for i, name in enumerate(class_names):
    print(
        f"{name} F1: {metrics['per_class_f1'][i]:.4f}, Precision: {metrics['per_class_precision'][i]:.4f}, Recall: {metrics['per_class_recall'][i]:.4f}"
    )

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=metrics["confusion_matrix"], display_labels=class_names
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Plot training loss and validation F1
plt.plot(train_losses, label="Training Loss")
plt.plot(val_f1s, label="Validation F1")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("training_plot.png")
plt.show()

# Visualize augmented sample
eb_idx = np.where(y == 2)[0][0]
x = torch.FloatTensor(X[eb_idx])
augmenter = F1OptimizedAugmentation(prob=1.0)
x_aug = augmenter(x, is_minority=True)
plt.figure(figsize=(12, 4))
for c, name in enumerate(["Flux", "Centroid", "Background"]):
    plt.subplot(1, 3, c + 1)
    plt.plot(x[:, c], label="Original")
    plt.plot(x_aug[:, c], label="Augmented", alpha=0.7)
    plt.title(f"{name} Channel")
    plt.legend()
plt.savefig("augmented_sample.png")
plt.show()
