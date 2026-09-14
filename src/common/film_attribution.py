"""
FiLM Attribution Analysis
Extracts the γ (gamma) and β (beta) values of the FiLM layer to analyze
how clinical variables modulate image features
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "data", "AIS_Surgery_clean.csv")
CACHE_PATH  = os.path.join(BASE_DIR, "cache", "scoliosis_lumbar_matched_data_512.pkl")
CKPT_PATTERN = os.path.join(
    BASE_DIR,
    "experiments_lumbar/scoliosis_structural/cv5_20260330_213940_512",
    "lumbar_only__film__resnet101_tv_in1k/fold{fold}/checkpoints/best_auroc.pth"
)
OUT_DIR = os.path.join(BASE_DIR, "test_results", "film_attribution")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLINICAL_NAMES = ["Sex", "Age", "Lumbar_Cobb", "L1_S1_Lordosis"]

# ─────────────────────────────────────────────
# Model / data loading utilities
# ─────────────────────────────────────────────
from dataset import ScoliosisLumbarDataset
from model import ScoliosisLumbarModel


def load_model(fold: int) -> tuple:
    ckpt_path = CKPT_PATTERN.format(fold=fold)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]

    model = ScoliosisLumbarModel(
        pa_backbone=cfg.get("pa_backbone", "resnet101.tv_in1k"),
        lat_backbone=cfg.get("lat_backbone", "resnet101.tv_in1k"),
        clinical_dim=cfg.get("clinical_dim", 4),
        hidden_dim=cfg.get("hidden_dim", 256),
        fusion_type=cfg.get("fusion_type", "film"),
        dropout_p=cfg.get("dropout_p", 0.4),
    )
    state = ckpt["model_state_dict"]
    # Strip DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────
# FiLM gamma/beta extraction (hook)
# ─────────────────────────────────────────────
def extract_film_params(model, dataset, test_indices):
    """Extract gamma and beta vectors for the test-set patients."""
    film_outputs = {}

    def hook_fn(module, inp, out):
        # Take the film_gen output (concatenated gamma_beta vector) directly,
        # rather than the modulated feature returned by FiLMFusion.forward
        pass

    # Register hook on film_gen
    gamma_list, beta_list = [], []

    def film_gen_hook(module, inp, out):
        gb = out.detach().cpu()
        half = gb.shape[1] // 2
        gamma_list.append(gb[:, :half])
        beta_list.append(gb[:, half:])

    handle = model.fusion.film_gen.register_forward_hook(film_gen_hook)

    patient_ids, labels, probs = [], [], []
    clin_raw = []

    from torch.utils.data import DataLoader, Subset
    subset = Subset(dataset, test_indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=True,
                        collate_fn=getattr(dataset, "_collate_fn", None))

    with torch.no_grad():
        for batch in loader:
            # dataset returns (dict, label) tuple
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_dict, batch_labels = batch
            else:
                batch_dict = batch
                batch_labels = batch_dict.pop("label")

            pa   = batch_dict["pa"].to(DEVICE)
            lat  = batch_dict["lat"].to(DEVICE)
            clin = batch_dict["clinical"].to(DEVICE)
            logit = model(pa, lat, clin)
            prob  = torch.sigmoid(logit).cpu().numpy().flatten()
            probs.extend(prob.tolist())
            pids = batch_dict["patient_id"]
            patient_ids.extend(pids if isinstance(pids, list) else [pids])
            if isinstance(batch_labels, torch.Tensor):
                labels.extend(batch_labels.cpu().numpy().flatten().tolist())
            else:
                labels.extend(list(batch_labels))
            clin_raw.append(batch_dict["clinical"].cpu().numpy())

    handle.remove()

    gamma = torch.cat(gamma_list, dim=0).numpy()  # (N, 4096)
    beta  = torch.cat(beta_list,  dim=0).numpy()  # (N, 4096)
    clin_arr = np.vstack(clin_raw)                 # (N, 4)

    return {
        "gamma": gamma, "beta": beta,
        "clinical": clin_arr,
        "patient_ids": patient_ids,
        "labels": np.array(labels),
        "probs": np.array(probs),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
all_gamma, all_beta, all_clin, all_labels, all_probs, all_pids = [], [], [], [], [], []

import pickle
cache = pickle.load(open(CACHE_PATH, "rb"))
matched_ids = set(int(k) for k in cache.keys())
clin_df = pd.read_csv(CSV_PATH)
clin_df = clin_df[clin_df["ID"].isin(matched_ids)].reset_index(drop=True)

for fold in range(1, 6):
    print(f"Processing fold {fold}...", end=" ", flush=True)
    ckpt_path = CKPT_PATTERN.format(fold=fold)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]

    dataset = ScoliosisLumbarDataset(
        data_dir=os.path.join(BASE_DIR, "data", "matched_data"),
        label_csv=CSV_PATH,
        clinical_set=cfg.get("clinical_set", "lumbar_only"),
        img_size=cfg.get("img_size", 512),
        mode="test",
        cache_dir=os.path.join(BASE_DIR, "cache"),
        use_cache=True,
    )

    test_indices = cfg.get("test_indices", [])
    if not test_indices:
        print("no test_indices, skipping")
        continue

    model, _ = load_model(fold)
    res = extract_film_params(model, dataset, test_indices)

    all_gamma.append(res["gamma"])
    all_beta.append(res["beta"])
    all_clin.append(res["clinical"])
    all_labels.append(res["labels"])
    all_probs.append(res["probs"])
    all_pids.extend(res["patient_ids"])
    print(f"N={len(res['labels'])}")

gamma_all = np.vstack(all_gamma)  # (155, 4096)
beta_all  = np.vstack(all_beta)
clin_all  = np.vstack(all_clin)   # (155, 4): Sex, Age, Lumbar_Cobb, L1_S1_Lordosis
labels_all = np.concatenate(all_labels)
probs_all  = np.concatenate(all_probs)

print(f"\nTotal gamma shape: {gamma_all.shape}")
print(f"Total beta  shape: {beta_all.shape}")

# ─────────────────────────────────────────────
# Analysis 1: channel-wise mean & std of γ/β (overall model behavior)
# ─────────────────────────────────────────────
gamma_mean_per_channel = gamma_all.mean(axis=0)   # (4096,)
beta_mean_per_channel  = beta_all.mean(axis=0)

gamma_abs_mean = np.abs(gamma_all).mean(axis=0)
beta_abs_mean  = np.abs(beta_all).mean(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].plot(np.sort(gamma_mean_per_channel), lw=0.8, color="#E53935")
axes[0,0].set_title("γ (gamma) — Sorted Channel Means", fontsize=11)
axes[0,0].set_xlabel("Channel rank"); axes[0,0].set_ylabel("Mean γ value")
axes[0,0].axhline(0, color="gray", lw=0.8, ls="--")

axes[0,1].plot(np.sort(beta_mean_per_channel), lw=0.8, color="#1E88E5")
axes[0,1].set_title("β (beta) — Sorted Channel Means", fontsize=11)
axes[0,1].set_xlabel("Channel rank"); axes[0,1].set_ylabel("Mean β value")
axes[0,1].axhline(0, color="gray", lw=0.8, ls="--")

axes[1,0].hist(gamma_abs_mean, bins=60, color="#E53935", alpha=0.7, edgecolor="white")
axes[1,0].set_title("|γ| Absolute Mean Distribution", fontsize=11)
axes[1,0].set_xlabel("|γ| magnitude"); axes[1,0].set_ylabel("# channels")

axes[1,1].hist(beta_abs_mean, bins=60, color="#1E88E5", alpha=0.7, edgecolor="white")
axes[1,1].set_title("|β| Absolute Mean Distribution", fontsize=11)
axes[1,1].set_xlabel("|β| magnitude"); axes[1,1].set_ylabel("# channels")

plt.suptitle("FiLM Parameters: Channel-wise Statistics", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "film_channel_stats.png"), dpi=150, bbox_inches="tight")
plt.close()
print("film_channel_stats.png saved")

# ─────────────────────────────────────────────
# Analysis 2: Spearman correlation between each clinical variable and γ/β
# (find top-correlated channels per clinical variable)
# ─────────────────────────────────────────────
corr_rows = []
for ci, cname in enumerate(CLINICAL_NAMES):
    clin_vec = clin_all[:, ci]

    # Spearman corr with each channel
    gamma_corrs = np.array([
        stats.spearmanr(clin_vec, gamma_all[:, ch]).statistic
        for ch in range(gamma_all.shape[1])
    ])
    beta_corrs = np.array([
        stats.spearmanr(clin_vec, beta_all[:, ch]).statistic
        for ch in range(beta_all.shape[1])
    ])

    # Mean |corr| across all channels → "influence" of the clinical variable
    gamma_influence = np.nanmean(np.abs(gamma_corrs))
    beta_influence  = np.nanmean(np.abs(beta_corrs))
    gamma_top_corr  = np.nanmax(np.abs(gamma_corrs))
    beta_top_corr   = np.nanmax(np.abs(beta_corrs))

    corr_rows.append({
        "clinical_var": cname,
        "gamma_mean_abs_corr": round(gamma_influence, 4),
        "gamma_max_abs_corr": round(gamma_top_corr, 4),
        "beta_mean_abs_corr": round(beta_influence, 4),
        "beta_max_abs_corr": round(beta_top_corr, 4),
    })
    print(f"  {cname}: γ mean|r|={gamma_influence:.4f}  β mean|r|={beta_influence:.4f}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(OUT_DIR, "film_clinical_influence.csv"), index=False)

# ─────────────────────────────────────────────
# Analysis 3: bar plot of influence per clinical variable
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(CLINICAL_NAMES))
w = 0.35

axes[0].bar(x, corr_df["gamma_mean_abs_corr"], width=0.5, color="#E53935", alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(CLINICAL_NAMES, fontsize=11)
axes[0].set_title("γ (scale): Mean |Spearman r|\nper Clinical Variable", fontsize=12)
axes[0].set_ylabel("Mean |Spearman r| across 4096 channels")

axes[1].bar(x, corr_df["beta_mean_abs_corr"], width=0.5, color="#1E88E5", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(CLINICAL_NAMES, fontsize=11)
axes[1].set_title("β (shift): Mean |Spearman r|\nper Clinical Variable", fontsize=12)
axes[1].set_ylabel("Mean |Spearman r| across 4096 channels")

plt.suptitle("FiLM Attribution: Clinical Variable Influence on Image Features", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "film_attribution_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("film_attribution_bar.png saved")

# ─────────────────────────────────────────────
# Analysis 4: compare patient-level mean of γ/β by label
# ─────────────────────────────────────────────
gamma_patient_mean = gamma_all.mean(axis=1)   # (N,) — per-patient gamma mean
beta_patient_mean  = beta_all.mean(axis=1)
gamma_patient_std  = gamma_all.std(axis=1)
beta_patient_std   = beta_all.std(axis=1)

plot_df = pd.DataFrame({
    "label": labels_all,
    "prob": probs_all,
    "gamma_mean": gamma_patient_mean,
    "beta_mean":  beta_patient_mean,
    "gamma_std":  gamma_patient_std,
    "beta_std":   beta_patient_std,
    "Lumbar_Cobb": clin_all[:, 2],
    "L1_S1_Lordosis": clin_all[:, 3],
})
plot_df["group"] = plot_df["label"].map({1: "Structural", 0: "Non-structural"})

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for ax, col, color, title in [
    (axes[0,0], "gamma_mean", "#E53935", "γ Mean per Patient"),
    (axes[0,1], "beta_mean",  "#1E88E5", "β Mean per Patient"),
    (axes[1,0], "gamma_std",  "#E53935", "γ Std per Patient"),
    (axes[1,1], "beta_std",   "#1E88E5", "β Std per Patient"),
]:
    sns.boxplot(data=plot_df, x="group", y=col,
                palette={"Structural": "#F44336", "Non-structural": "#42A5F5"},
                ax=ax, width=0.4)
    sns.stripplot(data=plot_df, x="group", y=col,
                  color="black", alpha=0.3, size=3, jitter=True, ax=ax)
    # Mann-Whitney U
    g1 = plot_df[plot_df["label"]==1][col].values
    g0 = plot_df[plot_df["label"]==0][col].values
    _, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    ax.set_title(f"{title}\n(p={p:.4f})", fontsize=11)
    ax.set_xlabel("")

plt.suptitle("FiLM Parameters: Structural vs Non-structural", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "film_label_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("film_label_comparison.png saved")

# ─────────────────────────────────────────────
# Analysis 5: Lumbar_Cobb vs gamma_mean scatter
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (xcol, xlabel) in zip(axes, [
    ("Lumbar_Cobb", "Lumbar Cobb Angle (°)"),
    ("L1_S1_Lordosis", "L1-S1 Lordosis (°)"),
]):
    sc = ax.scatter(plot_df[xcol], plot_df["gamma_mean"],
                    c=plot_df["prob"], cmap="RdYlGn",
                    s=50, alpha=0.8, edgecolors="gray", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Predicted Probability")
    r, p = stats.spearmanr(plot_df[xcol].dropna(), plot_df.loc[plot_df[xcol].notna(), "gamma_mean"])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("γ Mean (patient-level)", fontsize=11)
    ax.set_title(f"{xlabel} vs γ Mean\nSpearman r={r:.3f}, p={p:.4f}", fontsize=11)

plt.suptitle("FiLM γ vs Key Clinical Variables", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "film_scatter_clinical.png"), dpi=150, bbox_inches="tight")
plt.close()
print("film_scatter_clinical.png saved")

# ─────────────────────────────────────────────
# Final summary output
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FiLM ATTRIBUTION SUMMARY")
print("="*60)
print(corr_df.to_string(index=False))
print(f"\nResults saved to: {OUT_DIR}")
