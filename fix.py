import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import gaussian_filter1d
import numpy as np
import random
import os
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
# dataset_path = "final_dataset.npz"
dataset_path = "combined_dataset.npz"
try:
    data_npz = np.load(dataset_path, allow_pickle=True)
    print("Available keys in .npz file:", list(data_npz.keys()))
    X = data_npz["data"]
    y = data_npz["label"]
except FileNotFoundError:
    print(f"Error: Dataset file not found at {dataset_path}")
    raise
except KeyError:
    print("Error: 'data' or 'label' keys not found in .npz file")
    raise

# Verify shapes and class distribution
print("Data shape:", X.shape)
print("Label shape:", y.shape)
print("Class counts:", np.bincount(y))

# Validate shapes
if X.shape != (2891, 1000, 3) or y.shape != (2891,):
    raise ValueError(
        f"Expected data shape (2891, 1000, 3) and label shape (2891,), got {X.shape} and {y.shape}"
    )

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training class counts:", np.bincount(y_train))
print("Validation class counts:", np.bincount(y_val))


# Balanced Augmentation
class BalancedAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def time_warp(self, x):
        if random.random() > self.prob:
            return x
        sigma = 0.02  # Much smaller warping
        seq_len = x.shape[0]
        tt = torch.arange(seq_len).float()
        warp = torch.normal(0, sigma, (seq_len,)).cumsum(0)
        warp = (warp - warp.mean()) / (warp.std() + 1e-8) * sigma
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

    def add_noise(self, x):
        if random.random() > self.prob:
            return x
        noise_level = 0.005  # Very small noise
        noise = torch.normal(0, noise_level, x.shape)
        return x + noise

    def magnitude_scale(self, x):
        if random.random() > self.prob:
            return x
        scale = torch.normal(1.0, 0.02, (1, x.shape[1]))  # Small scaling
        return x * scale

    def __call__(self, x):
        x = self.time_warp(x)
        x = self.add_noise(x)
        x = self.magnitude_scale(x)
        return x


# Balanced Dataset
class BalancedDataset(Dataset):
    def __init__(self, data, labels, normalize=True, augment=False):
        self.original_data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.data = (
            self._normalize(self.original_data)
            if normalize
            else self.original_data.clone()
        )
        if augment:
            self.augmenter = BalancedAugmentation(prob=0.3)

    def _normalize(self, data):
        # Simple z-score normalization per channel
        normed = data.clone()
        for c in range(data.shape[2]):
            channel_data = data[:, :, c]
            mean = channel_data.mean()
            std = channel_data.std()
            normed[:, :, c] = (channel_data - mean) / (std + 1e-8)
        return normed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        y = self.labels[idx]
        if self.augment and random.random() < 0.3:  # Lower augmentation rate
            x = self.augmenter(x)
        return x, y


# Simplified CNN-LSTM Model
class SimplifiedCNNLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 from bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(
                module.weight, mode="fan_in", nonlinearity="relu"
            )

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, 128, seq_len/8)

        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch_size, seq_len/8, 128)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        x = (
            h_n[-1]
            if not self.lstm.bidirectional
            else torch.cat([h_n[-2], h_n[-1]], dim=1)
        )

        # Classification
        x = self.classifier(x)
        return x


# Balanced Loss Function
class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return nn.functional.cross_entropy(
            input, target, weight=self.weight, label_smoothing=self.label_smoothing
        )


# More balanced sampling
class_counts = np.bincount(y_train)
# Moderate class balancing - not extreme
sample_weights = 1.0 / (class_counts + 1e-8)
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
sample_weights = sample_weights[y_train]

# Cap the maximum weight to prevent extreme imbalance
max_weight = sample_weights.mean() * 3
sample_weights = np.clip(sample_weights, None, max_weight)

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(y_train), replacement=True
)

# DataLoaders
train_dataset = BalancedDataset(X_train, y_train, normalize=True, augment=True)
val_dataset = BalancedDataset(X_val, y_val, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model
model = SimplifiedCNNLSTM(input_channels=3, num_classes=3).to(device)

# More balanced class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
# Moderate the extreme weights
class_weights = np.clip(class_weights, 0.5, 3.0)  # Cap weights
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class weights:", class_weights)

# Loss and optimizer
criterion = BalancedCrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

# Training loop
num_epochs = 150
best_f1 = 0
patience = 15
counter = 0
train_losses, val_f1s = [], []

print("Starting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        # Debug: Print prediction distribution every 50 batches
        if batch_idx % 50 == 0:
            with torch.no_grad():
                _, pred = outputs.max(1)
                pred_counts = torch.bincount(pred, minlength=3)
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx}: Predictions: {pred_counts.cpu().numpy()}"
                )

    scheduler.step()
    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    preds, targets = [], []
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            val_loss += criterion(outputs, y).item()

            _, pred = outputs.max(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    val_f1 = f1_score(targets, preds, average="macro")
    per_class_f1 = f1_score(targets, preds, average=None)
    val_f1s.append(val_f1)
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
    print(
        f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
    )
    print(f"Val F1: {val_f1:.4f}")
    print(
        f"Per-Class F1: Noise={per_class_f1[0]:.4f}, Planetary={per_class_f1[1]:.4f}, EB={per_class_f1[2]:.4f}"
    )
    print(f"Prediction counts: {np.bincount(preds, minlength=3)}")
    print(f"True counts: {np.bincount(targets, minlength=3)}")
    print("-" * 50)

    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"New best F1: {best_f1:.4f}")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break


# Final evaluation
def calculate_comprehensive_metrics(targets, preds):
    macro_f1 = f1_score(targets, preds, average="macro")
    per_class_f1 = f1_score(targets, preds, average=None)
    per_class_precision = precision_score(targets, preds, average=None, zero_division=0)
    per_class_recall = recall_score(targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds)

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
    }


# Load best model for final evaluation
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
model.eval()

# Final validation predictions
final_preds, final_targets = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, pred = outputs.max(1)
        final_preds.extend(pred.cpu().numpy())
        final_targets.extend(y.cpu().numpy())

metrics = calculate_comprehensive_metrics(final_targets, final_preds)
class_names = ["Noise", "Planetary Transit", "Eclipsing Binary"]

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Final Macro F1: {metrics['macro_f1']:.4f}")
print(f"Best F1 during training: {best_f1:.4f}")

for i, name in enumerate(class_names):
    print(f"{name}:")
    print(f"  F1: {metrics['per_class_f1'][i]:.4f}")
    print(f"  Precision: {metrics['per_class_precision'][i]:.4f}")
    print(f"  Recall: {metrics['per_class_recall'][i]:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=metrics["confusion_matrix"], display_labels=class_names
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_f1s, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Save results
torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
history = {"train_losses": train_losses, "val_f1s": val_f1s}
np.savez(os.path.join(output_dir, "training_history.npz"), **history)

print(f"\nTraining complete. Best F1: {best_f1:.4f}")
print(f"Outputs saved to: {output_dir}")
