"""Plot feature distributions by ROI class (soma/dend/artefact).

Loads the /data/s2p/ legacy data, extracts the 13-feature set, and produces
boxplots for each feature split by the 3 classes. Saves to
reports/roi_features/feature_boxplots.png.

Usage
-----
    python scripts/plot_roi_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

S2P_ROOT = Path("/data/s2p")
OUT_DIR = Path("reports/roi_features")
LABEL_NAMES = {0: "artefact", 1: "soma", 2: "dend"}


def load_all() -> pd.DataFrame:
    """Load features + labels for all sessions."""
    rows = []
    for session_dir in sorted(S2P_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        soma_dir = session_dir / "suite2p_soma" / "plane0"
        dend_dir = session_dir / "suite2p_dend" / "plane0"
        if not soma_dir.exists() or not dend_dir.exists():
            continue

        ic_soma = np.load(soma_dir / "iscell.npy")
        ic_dend = np.load(dend_dir / "iscell.npy")

        n_soma = int((ic_soma[:, 0] == 1).sum())
        n_dend = int((ic_dend[:, 0] == 1).sum())
        if n_soma + n_dend == 0:
            continue

        n_rois = len(ic_soma)
        labels = np.zeros(n_rois, dtype=np.int64)
        labels[ic_soma[:, 0] == 1] = 1
        labels[ic_dend[:, 0] == 1] = 2

        stat = list(np.load(soma_dir / "stat.npy", allow_pickle=True))
        F = np.load(soma_dir / "F.npy").astype(np.float32)
        Fneu = np.load(soma_dir / "Fneu.npy").astype(np.float32)
        ops = np.load(soma_dir / "ops.npy", allow_pickle=True).item()
        fps = float(ops.get("fs", 9.6))

        features = extract_soma_features(stat, F, Fneu, fps=fps)
        features["label"] = [LABEL_NAMES[l] for l in labels]
        features["session"] = session_dir.name
        rows.append(features)

        print(f"  {session_dir.name}: {n_rois} ROIs ({n_soma} soma, {n_dend} dend)")

    return pd.concat(rows, ignore_index=True)


def plot_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    """One boxplot per feature, colored by class."""
    features = list(FEATURE_COLUMNS)
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    class_order = ["soma", "dend", "artefact"]
    colors = {"soma": "#2ecc71", "dend": "#e74c3c", "artefact": "#95a5a6"}

    for i, feat in enumerate(features):
        ax = axes[i]
        data = [df.loc[df["label"] == cls, feat].dropna().values for cls in class_order]
        bp = ax.boxplot(
            data,
            labels=class_order,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            medianprops=dict(color="black", linewidth=1.5),
            widths=0.6,
        )
        for patch, cls in zip(bp["boxes"], class_order):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.7)

        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)

        # Add counts
        for j, cls in enumerate(class_order):
            n = len(data[j])
            ax.text(j + 1, ax.get_ylim()[0], f"n={n}", ha="center", va="top", fontsize=7)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Feature distributions by ROI class (n={len(df)}: "
        f"{(df['label']=='soma').sum()} soma, "
        f"{(df['label']=='dend').sum()} dend, "
        f"{(df['label']=='artefact').sum()} artefact)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


def plot_pca(df: pd.DataFrame, out_dir: Path) -> None:
    """Z-score features, run PCA, plot variance explained and PC1 vs PC2."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = list(FEATURE_COLUMNS)
    X = df[features].copy()

    # Fill NaN with median, then z-score
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    class_order = ["soma", "dend", "artefact"]
    colors = {"soma": "#2ecc71", "dend": "#e74c3c", "artefact": "#95a5a6"}

    # --- Variance explained ---
    fig, ax = plt.subplots(figsize=(10, 4))
    n_comp = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_, color="#3498db", alpha=0.7, label="Individual")
    ax.plot(range(1, n_comp + 1), cumvar, "o-", color="#e74c3c", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Variance Explained")
    ax.legend()
    ax.set_xticks(range(1, n_comp + 1))
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=150, bbox_inches="tight")
    print(f"Saved PCA variance to {out_dir / 'pca_variance.png'}")

    # Print top loadings for PC1 and PC2
    for pc_idx in range(2):
        loadings = pca.components_[pc_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        print(f"\nPC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]:.1%} var) top loadings:")
        for j in sorted_idx[:8]:
            print(f"  {features[j]:>25s}: {loadings[j]:+.3f}")

    # --- PC1 vs PC2 scatter ---
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in class_order:
        mask = df["label"].values == cls
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[cls], label=f"{cls} (n={mask.sum()})",
            alpha=0.4 if cls == "artefact" else 0.7,
            s=10 if cls == "artefact" else 25,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: PC1 vs PC2 colored by ROI class")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter.png", dpi=150, bbox_inches="tight")
    print(f"Saved PCA scatter to {out_dir / 'pca_scatter.png'}")


def plot_correlation_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """Plot feature-feature correlation matrix (Spearman)."""
    features = list(FEATURE_COLUMNS)
    X = df[features].copy().fillna(df[features].median())
    corr = X.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(features, fontsize=8)

    # Annotate cells
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    ax.set_title("Feature Correlation Matrix (Spearman)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation matrix to {out_path}")

    # Print highly correlated pairs (|r| > 0.7)
    print("\nHighly correlated pairs (|r| > 0.7):")
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.values[i, j]
            if abs(r) > 0.7:
                print(f"  {features[i]:>25s} vs {features[j]:<25s}: r={r:+.3f}")


def main() -> int:
    print("Loading data...")
    df = load_all()
    print(f"\nTotal: {len(df)} ROIs")
    print(df["label"].value_counts().to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_boxplots(df, OUT_DIR / "feature_boxplots.png")
    plot_pca(df, OUT_DIR)
    plot_correlation_matrix(df, OUT_DIR / "correlation_matrix.png")

    # Summary stats per class
    print("\n" + "=" * 70)
    print("Feature summary (median) by class:")
    print("=" * 70)
    summary = df.groupby("label")[list(FEATURE_COLUMNS)].median()
    print(summary.round(3).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
