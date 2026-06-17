import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
import time
import os
import modal


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


# Set up Modal app and image
app = modal.App("maybe")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "scikit-learn", "matplotlib", "numpy")
    .add_local_file(
        "combined_dataset.npz", remote_path="/root/combined_dataset.npz", copy=True
    )
    .add_local_file("maybe.py", remote_path="/root/maybe.py", copy=True)
    .run_commands(
        "ln -s /root/maybe.py /root/lib/python*/site-packages/"
    )  # Make maybe.py importable
)


@app.function(gpu="A10G", timeout=14400, image=image)  # 4 hours in seconds
def train():
    # Set up device and output directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "/root/outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # Load and process data
    try:
        data_npz = np.load("/root/combined_dataset.npz", allow_pickle=True)
        print("Available keys in .npz file:", list(data_npz.keys()))
    except FileNotFoundError:
        print("Error: combined_dataset.npz not found in /root/")
        raise
    X = data_npz["data"]
    y = data_npz["label"]
    print("Data shape:", X.shape)
    print("Label shape:", y.shape)
    print("Class counts:", np.bincount(y))

    if X.shape != (2891, 1000, 3) or y.shape != (2891,):
        raise ValueError(
            f"Expected data shape (2891, 1000, 3) and label shape (2891,), got {X.shape} and {y.shape}"
        )

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training class counts:", np.bincount(y_train))
    print("Validation class counts:", np.bincount(y_val))

    # Import all class definitions from maybe.py
    from maybe import (
        F1OptimizedAugmentation,
        AttentiveMultiScaleCNN,
        EnhancedPositionalEncoding,
        F1OptimizedTransformer,
        FocalLoss,
        F1FocusedDataset,
    )

    # Create datasets and loaders
    train_dataset = F1FocusedDataset(X_train, y_train, normalize=True, augment=True)
    val_dataset = F1FocusedDataset(X_val, y_val, normalize=True, augment=False)

    # Set up sampling weights
    class_counts = np.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    sample_weights[y_train == 2] *= 3.0
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_train), replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    model = F1OptimizedTransformer(
        input_channels=3,
        d_model=384,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1536,
        dropout=0.3,
        num_classes=3,
    ).to(device)

    # Setup class weights and loss function
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights[2] *= 3.0
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights:", class_weights)

    # Create criterion, optimizer and scheduler
    criterion = FocalLoss(alpha=class_weights, gamma=4.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.03)
    scheduler = CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.001,
        step_size_up=4 * len(train_loader),
        mode="triangular",
    )

    # Training loop
    num_epochs = 150
    best_f1 = 0
    patience = 30
    counter = 0
    train_losses, val_f1s = [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0

        # Training phase
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_weights = torch.ones_like(y, dtype=torch.float, device=device)
            loss_weights[y == 2] = 3.0
            loss = (loss * loss_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation phase
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
        per_class_f1 = f1_score(targets, preds, average=None)
        val_f1s.append(val_f1)

        # Print epoch results
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s), "
            f"Loss: {train_losses[-1]:.4f}, Val F1: {val_f1:.4f}, "
            f"Per-Class F1: Noise={per_class_f1[0]:.4f}, "
            f"Planetary Transit={per_class_f1[1]:.4f}, "
            f"EB={per_class_f1[2]:.4f}"
        )
        print(f"Prediction counts: {np.bincount(preds, minlength=3)}")

        # Save best model and check early stopping
        print(f"Current F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
        else:
            counter += 1

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch+1}.pth")

        if counter >= patience:
            print("Early stopping")
            break

    # Final evaluation
    metrics = calculate_comprehensive_metrics(targets, preds)
    class_names = ["Noise", "Planetary Transit", "Eclipsing Binary"]

    print("Final Macro F1:", metrics["macro_f1"])
    for i, name in enumerate(class_names):
        print(
            f"{name} F1: {metrics['per_class_f1'][i]:.4f}, "
            f"Precision: {metrics['per_class_precision'][i]:.4f}, "
            f"Recall: {metrics['per_class_recall'][i]:.4f}"
        )

    # Save final visualizations and models
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"], display_labels=class_names
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # Plot training history
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_f1s, label="Validation F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{output_dir}/training_plot.png")
    plt.close()

    # Save final model and history
    torch.save(model.state_dict(), f"{output_dir}/final_model.pth")
    history = {"train_losses": train_losses, "val_f1s": val_f1s}
    np.savez(f"{output_dir}/training_history.npz", **history)
    np.savez(
        f"{output_dir}/class_weights.npz", class_weights=class_weights.cpu().numpy()
    )

    print(f"Training complete. All outputs saved to: {output_dir}")
    return best_f1


if __name__ == "__main__":
    app.run()
