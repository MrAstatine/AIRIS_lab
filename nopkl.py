# =====================================================================
#  Build refined_dataset.npz  (real + 1000 synthetic eclipsing binaries)
#  This version is safe for np.load without allow_pickle=True
# =====================================================================
import os, numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import periodogram

# 1 ───── paths ────────────────────────────────────────────────────────
drive_root = "."  # Use current directory or change as needed
orig_path = os.path.join(drive_root, "final_dataset.npz")
refined_path = os.path.join(drive_root, "refined_dataset.npz")

# 2 ───── load original ------------------------------------------------
data_npz = np.load(orig_path)
X_orig = data_npz["data"]  # (2591, 1000, 3)
y_orig = data_npz["label"]  # (2591,)
print("Original:", X_orig.shape, np.bincount(y_orig))

# 3 ───── get EB stats -------------------------------------------------
eb_idx = np.where(y_orig == 2)[0]
real_eb = X_orig[eb_idx]  # (198, 1000, 3)
flux_mean = real_eb[:, :, 0].mean()
flux_std = real_eb[:, :, 0].std()
centroid_mean = real_eb[:, :, 1].mean()
centroid_std = real_eb[:, :, 1].std()
background_m = real_eb[:, :, 2].mean()
background_s = real_eb[:, :, 2].std()

# crude period & depth distributions
periods = []
depths = []
for curve in real_eb[:, :, 0]:
    ac = np.correlate(curve - curve.mean(), curve - curve.mean(), mode="full")
    ac = ac[ac.size // 2 :]
    peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0]
    P = peaks[1] - peaks[0] if len(peaks) > 1 else 50
    periods.append(np.clip(P, 20, 200))
    depths.append(np.clip((curve.max() - curve.min()) / curve.max(), 0.01, 0.2))
P_mean, P_std = np.mean(periods), np.std(periods)
D_mean, D_std = np.mean(depths), np.std(depths)


# 4 ───── synthetic EB generator --------------------------------------
def synth_eb(n_samples=1000, T=1000):
    samples = []
    for _ in range(n_samples):
        t = np.arange(T)
        period = np.clip(np.random.normal(P_mean, 0.5 * P_std), 20, 200)
        depth = np.clip(np.random.normal(D_mean, D_std), 0.01, 0.2)
        dur = np.random.uniform(0.05, 0.15) * period
        phase = np.random.uniform(0, period)

        # flux channel
        flux = np.ones(T) * flux_mean
        for k in range(int(T / period) + 2):
            center = (k * period + phase) % T
            ing = center - dur / 2
            egr = center + dur / 2
            mask = (t >= ing) & (t <= egr)
            flux[mask] *= 1 - depth
        flux += np.random.normal(0, 0.05 * flux_std, T)
        flux = gaussian_filter1d(flux, sigma=1.5)

        # centroid + background = low-amp sinusoids with noise
        centroid = centroid_mean + centroid_std * np.sin(
            2 * np.pi * t / np.random.normal(110, 15)
        )
        background = background_m + background_s * np.sin(
            2 * np.pi * t / np.random.normal(200, 25)
        )
        sample = np.stack([flux, centroid, background], axis=-1)
        samples.append(sample.astype(np.float32))
    return np.array(samples)


X_syn = synth_eb(1000)  # (1000, 1000, 3)
y_syn = np.full(len(X_syn), 2, dtype=np.int64)
print("Synthetic:", X_syn.shape)


# 5 ───── periodogram features  (five numbers per sample) --------------
def periodogram_feats(data):
    feats = []
    for curve in data:
        f, p = periodogram(curve[:, 0], fs=1.0)
        idx = np.argsort(p)[-3:]  # three strongest freqs
        feat = [p.mean(), p.max(), p.var(), f[idx[1]], f[idx[2]]]
        feats.append(feat)
    return np.array(feats, dtype=np.float32)  # (N,5)


feat_orig = periodogram_feats(X_orig)  # (2591,5)
feat_syn = periodogram_feats(X_syn)  # (1000,5)


# ─ broadcast to time axis & append as channels
def attach_feats(X, feats):
    feats_rep = np.repeat(feats[:, None, :], X.shape[1], axis=1)  # (N,1000,5)
    return np.concatenate([X, feats_rep], axis=2)  # (N,1000,8)


X_orig_aug = attach_feats(X_orig, feat_orig)
X_syn_aug = attach_feats(X_syn, feat_syn)

# 6 ───── merge & save -------------------------------------------------
X_combined = np.concatenate([X_orig_aug, X_syn_aug], axis=0)
y_combined = np.concatenate([y_orig, y_syn], axis=0)
np.savez(refined_path, data=X_combined, label=y_combined)

print("\nRefined file written →", refined_path)
print("New shapes:", X_combined.shape, y_combined.shape)
print("New class counts:", np.bincount(y_combined))
