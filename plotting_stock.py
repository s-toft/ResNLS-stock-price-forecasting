import numpy as np
import matplotlib.pyplot as plt

# === Load saved arrays ===
base_leaky_pred  = np.load("baseline_leaky_preds.npy")
base_leaky_true  = np.load("baseline_leaky_targets.npy")

ext_leaky_pred   = np.load("extended_leaky_preds.npy")
ext_leaky_true   = np.load("extended_leaky_targets.npy")

base_clean_pred  = np.load("baseline_clean_preds.npy")
base_clean_true  = np.load("baseline_clean_targets.npy")

ext_clean_pred   = np.load("extended_clean_preds.npy")
ext_clean_true   = np.load("extended_clean_targets.npy")

# Sanity check: all test arrays should have the same length
assert len(base_leaky_true) == len(ext_leaky_true) == len(base_clean_true) == len(ext_clean_true), "Mismatched test lengths."

# === Global y-limits across all panels ===
all_vals = np.concatenate([
    base_leaky_true, base_leaky_pred,
    ext_leaky_true,  ext_leaky_pred,
    base_clean_true, base_clean_pred,
    ext_clean_true,  ext_clean_pred
])
ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
pad = 0.05 * (ymax - ymin + 1e-9)
ylims = (ymin - pad, ymax + pad)

# === Plot helper ===
def plot_panel(ax, y_true, y_pred, title):
    x = np.arange(len(y_true))
    ax.plot(x, y_true, label="Actual (test)")
    ax.plot(x, y_pred, label="Predicted", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Close price")
    ax.set_ylim(*ylims)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

# === 2x2 composite ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

plot_panel(axes[0,0], base_leaky_true, base_leaky_pred,  "Baseline (close-only), LEAKY")
plot_panel(axes[0,1], base_clean_true, base_clean_pred,  "Baseline (close-only), CLEAN")
plot_panel(axes[1,0], ext_leaky_true,  ext_leaky_pred,   "Extended (OHLC), LEAKY")
plot_panel(axes[1,1], ext_clean_true,  ext_clean_pred,   "Extended (OHLC), CLEAN")

fig.suptitle("Predicted vs. Actual (Test Set)", y=1.02, fontsize=14)
plt.savefig("fig_all_four_test_only.png", dpi=300, bbox_inches="tight")
plt.show()

# Load again (or reuse from above)
base_leaky_pred  = np.load("baseline_leaky_preds.npy")
base_leaky_true  = np.load("baseline_leaky_targets.npy")
ext_leaky_pred   = np.load("extended_leaky_preds.npy")
ext_leaky_true   = np.load("extended_leaky_targets.npy")

base_clean_pred  = np.load("baseline_clean_preds.npy")
base_clean_true  = np.load("baseline_clean_targets.npy")
ext_clean_pred   = np.load("extended_clean_preds.npy")
ext_clean_true   = np.load("extended_clean_targets.npy")

# --- Compute per-sample deltas ---
delta_leaky = np.abs(base_leaky_true - base_leaky_pred) - np.abs(ext_leaky_true - ext_leaky_pred)
delta_clean = np.abs(base_clean_true - base_clean_pred) - np.abs(ext_clean_true - ext_clean_pred)

# --- Bootstrap CI helper ---
def bootstrap_ci(x, n_boot=10000, alpha=0.05, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    n   = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return x.mean(), lo, hi

mean_L, lo_L, hi_L = bootstrap_ci(delta_leaky)
mean_C, lo_C, hi_C = bootstrap_ci(delta_clean)

print(f"Leaky ΔMAE mean: {mean_L:.2f}, 95% CI: [{lo_L:.2f}, {hi_L:.2f}]")
print(f"Clean ΔMAE mean: {mean_C:.2f}, 95% CI: [{lo_C:.2f}, {hi_C:.2f}]")

# --- Plot two histograms side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

def plot_delta(ax, delta, title, mean_val, lo, hi):
    ax.hist(delta, bins=30, alpha=0.85)
    ax.axvline(mean_val, linestyle="--", linewidth=2)
    ax.axvspan(lo, hi, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta$MAE (baseline $-$ extended)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

plot_delta(axes[0], delta_leaky, f"Leaky: mean={mean_L:.2f} [{lo_L:.2f}, {hi_L:.2f}]", mean_L, lo_L, hi_L)
plot_delta(axes[1], delta_clean, f"Clean: mean={mean_C:.2f} [{lo_C:.2f}, {hi_C:.2f}]", mean_C, lo_C, hi_C)

plt.savefig("fig_delta_mae_leaky_vs_clean.png", dpi=300, bbox_inches="tight")
plt.show()
