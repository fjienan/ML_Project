#!/usr/bin/env python3
"""DEC clustering for the breast cancer proteomics project."""

from __future__ import annotations

import copy
import os
import random
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


SEED = 42
K_VALUES = list(range(2, 9))
MAIN_K = 3
COMPARE_K = 4
PRETRAIN_EPOCHS = 500
DEC_EPOCHS = 350
LATENT_DIM = 10
HIDDEN_DIM = 32

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "code" / "clustering" / "results"
FIGURE_DIR = ROOT / "article" / "figures" / "clustering"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_dir() -> Path:
    required = [
        "77_cancer_proteomes_CPTAC_itraq.csv",
        "clinical_data_breast_cancer.csv",
        "PAM50_proteins.csv",
    ]
    for candidate in [ROOT / "code" / "preprocessing", ROOT / "code", ROOT / "questions" / "Course_Project_II"]:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError("Could not find the three required CSV data files.")


def format_sample_id(column_name: str) -> str:
    if "TCGA" not in column_name:
        return column_name
    return f"TCGA-{re.split(r'[_.]', column_name)[0]}"


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    d = data_dir()
    raw = pd.read_csv(d / "77_cancer_proteomes_CPTAC_itraq.csv", index_col=0)
    clinical = pd.read_csv(d / "clinical_data_breast_cancer.csv", index_col=0)
    pam50 = pd.read_csv(d / "PAM50_proteins.csv")

    expr = raw.drop(columns=["gene_symbol", "gene_name"], errors="ignore")
    expr = expr.rename(columns=format_sample_id)
    expr = expr.loc[:, ~expr.columns.duplicated()].T
    expr = expr.loc[[idx for idx in expr.index if idx in clinical.index]]
    merged = expr.merge(clinical[["PAM50 mRNA"]], left_index=True, right_index=True, how="inner")
    merged = merged.dropna(subset=["PAM50 mRNA"])

    pam50_ids = set(pam50["RefSeqProteinID"].dropna().astype(str))
    feature_cols = [col for col in expr.columns if col in pam50_ids]
    if len(merged) != 77:
        raise ValueError(f"Expected 77 tumor samples, got {len(merged)}.")
    if len(feature_cols) != 43:
        raise ValueError(f"Expected 43 matched PAM50 proteins, got {len(feature_cols)}.")

    x = merged[feature_cols].astype(float)
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    return merged, x, merged["PAM50 mRNA"].to_numpy(), feature_cols


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, LATENT_DIM))
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, input_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class DEC(nn.Module):
    def __init__(self, encoder: nn.Module, centers: np.ndarray, alpha: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        dist = torch.sum((z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q, z


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = (q**2) / torch.sum(q, dim=0, keepdim=True)
    return (weight.T / torch.sum(weight, dim=1)).T


def train_autoencoder(x: np.ndarray, device: torch.device) -> tuple[AutoEncoder, list[float]]:
    model = AutoEncoder(x.shape[1]).to(device)
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(PRETRAIN_EPOCHS):
        recon, _ = model(tensor)
        loss = F.mse_loss(recon, tensor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def train_dec(x: np.ndarray, pretrained: AutoEncoder, k: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[float]]:
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    pretrained.eval()
    with torch.no_grad():
        _, latent = pretrained(tensor)
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED + k).fit(latent.cpu().numpy())
    dec = DEC(copy.deepcopy(pretrained.encoder), km.cluster_centers_).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    losses = []
    for _ in range(DEC_EPOCHS):
        q, _ = dec(tensor)
        p = target_distribution(q).detach()
        loss = F.kl_div(torch.log(q.clamp_min(1e-10)), p, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    dec.eval()
    with torch.no_grad():
        q, z = dec(tensor)
    return torch.argmax(q, dim=1).cpu().numpy(), z.cpu().numpy(), losses


def metrics(method: str, k: int, x_for_sil: np.ndarray, labels: np.ndarray, y: np.ndarray) -> dict[str, float | str | int]:
    return {
        "method": method,
        "k": k,
        "Silhouette": silhouette_score(x_for_sil, labels),
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "Homogeneity": homogeneity_score(y, labels),
        "Completeness": completeness_score(y, labels),
        "V-measure": v_measure_score(y, labels),
    }


def save_line(df: pd.DataFrame, cols: list[str], title: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_values = df["k"].to_numpy()
    for col in cols:
        ax.plot(x_values, df[col].to_numpy(dtype=float), marker="o", linewidth=2, label=col)
    ax.set_title(title)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=300)
    plt.close(fig)


def save_loss(losses: list[float], title: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=300)
    plt.close(fig)


def save_scatter(embedding: np.ndarray, labels: np.ndarray, title: str, name: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    labels = np.array([str(v) for v in labels])
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=55, label=label, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged, x, y, feature_cols = load_dataset()
    print(f"Loaded matrix: {x.shape[0]} samples x {x.shape[1]} PAM50 protein features")

    kmeans_rows = []
    kmeans_labels = {}
    for k in K_VALUES:
        labels = KMeans(n_clusters=k, n_init=50, random_state=SEED + k).fit_predict(x)
        kmeans_labels[k] = labels
        kmeans_rows.append(metrics("KMeans", k, x, labels, y))
    kmeans_df = pd.DataFrame(kmeans_rows)
    kmeans_df.to_csv(RESULTS_DIR / "kmeans_by_k_metrics.csv", index=False, encoding="utf-8-sig")

    ae, ae_losses = train_autoencoder(x, device)
    dec_labels, dec_latent, dec_losses = train_dec(x, ae, MAIN_K, device)
    dec_labels4, dec_latent4, _ = train_dec(x, ae, COMPARE_K, device)

    summary = pd.DataFrame(
        [
            metrics("KMeans", MAIN_K, x, kmeans_labels[MAIN_K], y),
            metrics("KMeans", COMPARE_K, x, kmeans_labels[COMPARE_K], y),
            metrics("DEC", MAIN_K, x, dec_labels, y),
            metrics("DEC", COMPARE_K, x, dec_labels4, y),
        ]
    )
    summary.to_csv(RESULTS_DIR / "method_metrics_summary.csv", index=False, encoding="utf-8-sig")

    assignments = pd.DataFrame(index=merged.index)
    assignments.index.name = "SampleID"
    assignments["PAM50 mRNA"] = y
    assignments["KMeans_k3"] = kmeans_labels[MAIN_K]
    assignments["KMeans_k4"] = kmeans_labels[COMPARE_K]
    assignments["DEC_k3"] = dec_labels
    assignments["DEC_k4"] = dec_labels4
    assignments.to_csv(RESULTS_DIR / "cluster_assignments.csv", encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(RESULTS_DIR / "pam50_feature_columns.csv", index=False, encoding="utf-8-sig")

    table = pd.crosstab(pd.Series(y, name="PAM50"), pd.Series(dec_labels, name="DEC cluster"))
    table.to_csv(RESULTS_DIR / "pam50_dec_crosstab.csv", encoding="utf-8-sig")

    save_line(kmeans_df, ["Silhouette"], "K-means silhouette across k", "Silhouette", "kmeans_silhouette_curve.png")
    save_line(kmeans_df, ["ARI", "NMI", "Homogeneity"], "K-means external metrics across k", "Metric value", "kmeans_external_metrics.png")
    save_loss(ae_losses, "AutoEncoder pretraining loss", "MSE reconstruction loss", "dec_reconstruction_loss.png")
    save_loss(dec_losses, "DEC clustering KL loss (k=3)", "KL divergence", "dec_kl_loss.png")

    raw_pca = PCA(n_components=2, random_state=SEED).fit_transform(x)
    latent_pca = PCA(n_components=2, random_state=SEED).fit_transform(dec_latent)
    save_scatter(raw_pca, y, "PCA colored by PAM50", "pca_pam50_labels.png", "PC1", "PC2")
    save_scatter(raw_pca, [f"C{i}" for i in kmeans_labels[MAIN_K]], "PCA colored by K-means clusters", "pca_kmeans_clusters.png", "PC1", "PC2")
    save_scatter(raw_pca, [f"C{i}" for i in dec_labels], "PCA colored by DEC clusters", "pca_dec_clusters.png", "PC1", "PC2")
    save_scatter(latent_pca, y, "DEC latent space colored by PAM50", "latent_pam50_labels.png", "Latent PC1", "Latent PC2")
    save_scatter(latent_pca, [f"C{i}" for i in dec_labels], "DEC latent space colored by DEC clusters", "latent_dec_clusters.png", "Latent PC1", "Latent PC2")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("PAM50 subtype vs DEC cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pam50_dec_crosstab.png", dpi=300)
    plt.close(fig)

    order = np.lexsort((np.array([str(v) for v in y]), dec_labels))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    image = ax.imshow(x[order], aspect="auto", cmap="coolwarm", vmin=-2.8, vmax=2.8)
    ax.set_title("PAM50 protein expression sorted by DEC cluster")
    ax.set_xlabel("PAM50 protein feature")
    ax.set_ylabel("Samples sorted by cluster")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dec_expression_heatmap.png", dpi=300)
    plt.close(fig)

    plot_df = summary[summary["k"].isin([MAIN_K, COMPARE_K])]
    metric_cols = ["Silhouette", "ARI", "NMI", "Homogeneity", "V-measure"]
    labels = [f"{r.method} k={r.k}" for r in plot_df.itertuples()]
    xpos = np.arange(len(metric_cols))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.bar(xpos + (idx - 1.5) * width, [row[m] for m in metric_cols], width=width, label=labels[idx])
    ax.set_xticks(xpos)
    ax.set_xticklabels(metric_cols, rotation=25, ha="right")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "method_metrics_comparison.png", dpi=300)
    plt.close(fig)

    print(f"Saved results to {RESULTS_DIR}")
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
