import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import modal

# Create Modal app and configure image
app = modal.App("f1_optimized_transformer")

# Configure image with required packages
image = (
    modal.Image.debian_slim()
    .pip_install(["torch", "scikit-learn", "matplotlib", "numpy"])
    .add_local_file("merged_dataset.npz", "/root/merged_dataset.npz")
)


# Minimal stub for F1OptimizedAugmentation
class F1OptimizedAugmentation:
    def __init__(self, prob=0.95):
        self.prob = prob

    def __call__(self, x, is_minority=False):
        return x


# Minimal stub for F1FocusedDataset
class F1FocusedDataset(Dataset):
    def __init__(self, data, labels, normalize=True, augment=False, class_weights=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@app.function(image=image, gpu="any", timeout=1800)
def main(input_file):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load dataset ===
    data_npz = np.load("/root/merged_dataset.npz")
    X = data_npz["data"]
    y = data_npz["label"]
    print("Loaded merged_dataset.npz. Keys:", list(data_npz.keys()))
    print("Data shape:", X.shape)
    print("Label shape:", y.shape)
    print("Class counts:", np.bincount(y))

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
    train_loader = DataLoader(
        train_dataset, batch_size=16, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

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
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Training loop
    best_f1 = 0
    patience = 10
    counter = 0
    train_losses, val_f1s = [], []
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y_batch)
            loss_weights = torch.ones_like(y_batch, dtype=torch.float, device=device)
            loss_weights[y_batch == 2] = 2.0
            loss = (loss * loss_weights).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                out = model(x)
                _, pred = out.max(1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
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
        cm = confusion_matrix(targets, preds)
        return {
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm,
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_f1s, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Training Progress")
    plt.savefig("training_plot.png")
    plt.show()

    # Visualize augmented sample
    eb_idx = np.where(y == 2)[0][0]
    x_sample = torch.FloatTensor(X[eb_idx])
    augmenter = F1OptimizedAugmentation(prob=1.0)
    x_aug = augmenter(x_sample, is_minority=True)
    plt.figure(figsize=(12, 4))
    for c, name in enumerate(["Flux", "Centroid", "Background"]):
        plt.subplot(1, 3, c + 1)
        plt.plot(x_sample[:, c], label="Original")
        plt.plot(x_aug[:, c], label="Augmented", alpha=0.7)
        plt.title(f"{name} Channel")
        plt.legend()
    plt.tight_layout()
    plt.savefig("augmented_sample.png")
    plt.show()


# Modal Entrypoint
if __name__ == "__main__":
    with app.run():
        main.remote()


# AttentiveMultiScaleCNN
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
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
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
    def __init__(
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
        super().__init__()
        self.feature_extractor = AttentiveMultiScaleCNN(input_channels, d_model)
        self.pos_encoder = EnhancedPositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=False,
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
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
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


if __name__ == "__main__":
    with app.run():
        main.remote()
