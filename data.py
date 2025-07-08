import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seed
np.random.seed(42)

# Input/output paths
input_real = "final_dataset.npz"
input_synth = "synthetic_eb.npz"
output_dir = "."
output_path = os.path.join(output_dir, "combined_dataset.npz")

# Load real dataset
try:
    data_npz = np.load(input_real)
    X = data_npz["data"]
    y = data_npz["label"]
except FileNotFoundError:
    print(f"Error: Dataset file not found at {input_real}")
    raise
except KeyError:
    print("Error: 'data' or 'label' keys not found in .npz file")
    raise

# Verify shapes
if X.shape != (2591, 1000, 3) or y.shape != (2591,):
    raise ValueError(f"Expected data shape (2591, 1000, 3) and label shape (2591,), got {X.shape} and {y.shape}")

# Load synthetic EB data
try:
    synthetic_npz = np.load(input_synth)
    synthetic_X = synthetic_npz["data"]
    synthetic_y = synthetic_npz["label"]
except FileNotFoundError:
    print(f"Error: Synthetic data file not found at {input_synth}")
    raise
except KeyError:
    print("Error: 'data' or 'label' keys not found in synthetic .npz file")
    raise

# Verify synthetic data shape
if synthetic_X.shape != (300, 1000, 3) or synthetic_y.shape != (300,) or not np.all(synthetic_y == 2):
    raise ValueError(f"Expected synthetic data shape (300, 1000, 3) and labels (300,) all EB (2), got {synthetic_X.shape} and {synthetic_y.shape}")

# Combine real and synthetic data
X_combined = np.concatenate([X, synthetic_X], axis=0)
y_combined = np.concatenate([y, synthetic_y], axis=0)
print("Combined class counts:", np.bincount(y_combined))

# Train-validation split (optional, for info)
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
print("Training class counts:", np.bincount(y_train))
print("Validation class counts:", np.bincount(y_val))

# Save combined dataset (no pickle, only arrays)
np.savez(output_path, data=X_combined, label=y_combined)
print(f"Combined dataset saved to {output_path}")
