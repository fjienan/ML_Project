#!/usr/bin/env python3
"""Own problem: predict lymph node involvement from proteomic profiles."""

from __future__ import annotations

import copy
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F


SEED = 42
N_SPLITS = 5
MISSING_THRESHOLD = 0.40
TOP_FEATURES = 500
DAE_EPOCHS = 350
MLP_EPOCHS = 280
PATIENCE = 45
DROPOUT = 0.30
NOISE_STD = 0.15

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "code" / "own_problem" / "results"
FIGURE_DIR = ROOT / "article" / "figures" / "own_problem"

METHOD_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "DAE_MLP": "Denoising AE + MLP",
}


@dataclass
class PreprocessState:
    selected_features: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(last, hidden), nn.ReLU(), nn.Dropout(DROPOUT)]
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_dir() -> Path:
    required = ["77_cancer_proteomes_CPTAC_itraq.csv", "clinical_data_breast_cancer.csv"]
    for candidate in [ROOT / "code" / "preprocessing", ROOT / "code", ROOT / "questions" / "Course_Project_II"]:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError("Could not find required CSV data files.")


def format_sample_id(column_name: str) -> str:
    if "TCGA" not in column_name:
        return column_name
    return f"TCGA-{re.split(r'[_.]', column_name)[0]}"


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    d = data_dir()
    raw = pd.read_csv(d / "77_cancer_proteomes_CPTAC_itraq.csv", index_col=0)
    clinical = pd.read_csv(d / "clinical_data_breast_cancer.csv", index_col=0)
    metadata = raw[["gene_symbol", "gene_name"]].copy()
    expr = raw.drop(columns=["gene_symbol", "gene_name"], errors="ignore")
    expr = expr.rename(columns=format_sample_id)
    expr = expr.loc[:, ~expr.columns.duplicated()].T
    expr = expr.loc[[idx for idx in expr.index if idx in clinical.index]]
    merged = expr.merge(clinical[["Node-Coded", "Node", "PAM50 mRNA"]], left_index=True, right_index=True, how="inner")
    merged = merged.dropna(subset=["Node-Coded"])
    y = (merged["Node-Coded"] == "Positive").astype(int).to_numpy()
    x = merged[expr.columns].astype(float)
    if len(x) != 77:
        raise ValueError(f"Expected 77 matched tumor samples, got {len(x)}.")
    return x, y, merged[["Node-Coded", "Node", "PAM50 mRNA"]].copy(), metadata


def fit_preprocess(x_train: pd.DataFrame) -> PreprocessState:
    kept = x_train.columns[x_train.isna().mean(axis=0) <= MISSING_THRESHOLD].tolist()
    filtered = x_train[kept]
    medians = filtered.median(axis=0)
    imputed = filtered.fillna(medians)
    selected = imputed.var(axis=0, ddof=0).sort_values(ascending=False).head(TOP_FEATURES).index.tolist()
    train_selected = imputed[selected]
    means = train_selected.mean(axis=0)
    stds = train_selected.std(axis=0, ddof=0).replace(0, 1.0)
    return PreprocessState(selected, medians[selected], means, stds)


def apply_preprocess(x: pd.DataFrame, state: PreprocessState) -> np.ndarray:
    selected = x[state.selected_features].fillna(state.medians)
    return ((selected - state.means) / state.stds).to_numpy(dtype=np.float32)


def split_inner(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def train_dae(x_train: np.ndarray, y_train: np.ndarray, device: torch.device, seed: int) -> tuple[DenoisingAutoEncoder, pd.DataFrame]:
    set_seed(seed)
    train_idx, val_idx = split_inner(y_train, seed)
    train_x = torch.tensor(x_train[train_idx], dtype=torch.float32, device=device)
    val_x = torch.tensor(x_train[val_idx], dtype=torch.float32, device=device)
    model = DenoisingAutoEncoder(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = PATIENCE
    history = []
    for epoch in range(1, DAE_EPOCHS + 1):
        model.train()
        noisy = train_x + torch.randn_like(train_x) * NOISE_STD
        recon, _ = model(noisy)
        loss = F.mse_loss(recon, train_x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(val_x)[0], val_x)
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": float(val_loss.detach().cpu())})
        if float(val_loss) < best_val - 1e-5:
            best_val = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def encode(model: DenoisingAutoEncoder, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(x, dtype=torch.float32, device=device))
    return z.cpu().numpy().astype(np.float32)


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, device: torch.device, seed: int) -> tuple[MLPClassifierTorch, pd.DataFrame]:
    set_seed(seed)
    train_idx, val_idx = split_inner(y_train, seed)
    train_x = torch.tensor(x_train[train_idx], dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train[train_idx], dtype=torch.float32, device=device)
    val_x = torch.tensor(x_train[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_train[val_idx], dtype=torch.float32, device=device)
    model = MLPClassifierTorch(x_train.shape[1], (16,)).to(device)
    pos = float(train_y.sum().item())
    neg = float(len(train_y) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = PATIENCE
    history = []
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        loss = loss_fn(model(train_x), train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y)
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": float(val_loss.detach().cpu())})
        if float(val_loss) < best_val - 1e-5:
            best_val = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_mlp(model: MLPClassifierTorch, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32, device=device))).cpu().numpy()


def metric_row(method: str, fold: int, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str | int]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "method": method,
        "fold": fold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    return centered @ vt[:2].T, explained[:2]


def plot_training(history: pd.DataFrame, title: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    epochs = history["epoch"].to_numpy()
    ax.plot(epochs, history["train_loss"].to_numpy(dtype=float), label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"].to_numpy(dtype=float), label="Validation", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=300)
    plt.close(fig)


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_NAMES:
        part = metrics_df[metrics_df["method"] == method]
        row: dict[str, float | str] = {"method": method}
        for metric in ["Accuracy", "Balanced Accuracy", "F1", "ROC-AUC", "PR-AUC"]:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(summary: pd.DataFrame, oof: pd.DataFrame, clinical: pd.DataFrame) -> None:
    counts = clinical["Node-Coded"].value_counts().reindex(["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Node-Coded label distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.6, str(int(v)), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "node_label_distribution.png", dpi=300)
    plt.close(fig)

    metrics = ["Balanced Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    x_pos = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for idx, method in enumerate(summary["method"]):
        row = summary[summary["method"] == method].iloc[0]
        ax.bar(x_pos + (idx - 1) * width, [row[f"{m}_mean"] for m in metrics], yerr=[row[f"{m}_std"] for m in metrics], width=width, label=METHOD_NAMES[method], capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_metrics_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for method in METHOD_NAMES:
        part = oof[oof["method"] == method]
        fpr, tpr, _ = roc_curve(part["y_true"], part["y_prob"])
        auc = roc_auc_score(part["y_true"], part["y_prob"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{METHOD_NAMES[method]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cross-validated ROC curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    main = oof[oof["method"] == "DAE_MLP"]
    cm = confusion_matrix(main["y_true"], main["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred Negative", "Pred Positive"], yticklabels=["True Negative", "True Positive"], ax=ax)
    ax.set_title("Denoising AE + MLP confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def run_cv(x_df: pd.DataFrame, y: np.ndarray, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metric_rows = []
    oof_rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_df, y), 1):
        print(f"Fold {fold}: train={len(train_idx)}, validation={len(val_idx)}")
        state = fit_preprocess(x_df.iloc[train_idx])
        x_train = apply_preprocess(x_df.iloc[train_idx], state)
        x_val = apply_preprocess(x_df.iloc[val_idx], state)
        y_train, y_val = y[train_idx], y[val_idx]

        logistic = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED + fold)
        logistic.fit(x_train, y_train)
        forest = RandomForestClassifier(n_estimators=500, class_weight="balanced", max_features="sqrt", min_samples_leaf=2, random_state=SEED + fold)
        forest.fit(x_train, y_train)
        dae, _ = train_dae(x_train, y_train, device, SEED + fold * 101)
        z_train = encode(dae, x_train, device)
        z_val = encode(dae, x_val, device)
        mlp, _ = train_mlp(z_train, y_train, device, SEED + fold * 211)

        predictions = {
            "LogisticRegression": logistic.predict_proba(x_val)[:, 1],
            "RandomForest": forest.predict_proba(x_val)[:, 1],
            "DAE_MLP": predict_mlp(mlp, z_val, device),
        }
        for method, prob in predictions.items():
            metric_rows.append(metric_row(method, fold, y_val, prob))
            for sample_id, truth, p in zip(x_df.index[val_idx], y_val, prob):
                oof_rows.append({"SampleID": sample_id, "fold": fold, "method": method, "y_true": int(truth), "y_prob": float(p), "y_pred": int(p >= 0.5)})
    return pd.DataFrame(metric_rows), pd.DataFrame(oof_rows)


def train_final(x_df: pd.DataFrame, y: np.ndarray, metadata: pd.DataFrame, device: torch.device) -> None:
    state = fit_preprocess(x_df)
    x_all = apply_preprocess(x_df, state)
    pd.DataFrame({"feature": state.selected_features}).join(metadata, on="feature").to_csv(RESULTS_DIR / "selected_features_final.csv", index=False, encoding="utf-8-sig")
    dae, dae_hist = train_dae(x_all, y, device, SEED + 9001)
    z_all = encode(dae, x_all, device)
    mlp, mlp_hist = train_mlp(z_all, y, device, SEED + 9011)
    dae_hist.to_csv(RESULTS_DIR / "final_dae_training_history.csv", index=False, encoding="utf-8-sig")
    mlp_hist.to_csv(RESULTS_DIR / "final_mlp_training_history.csv", index=False, encoding="utf-8-sig")
    plot_training(dae_hist, "Denoising AutoEncoder training curve", "MSE reconstruction loss", "dae_reconstruction_loss.png")
    plot_training(mlp_hist, "Latent MLP training curve", "Binary cross-entropy loss", "mlp_training_curve.png")

    emb, exp = pca_2d(z_all)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for label, color, value in [("Negative", "#4C78A8", 0), ("Positive", "#F58518", 1)]:
        mask = y == value
        ax.scatter(emb[mask, 0], emb[mask, 1], color=color, label=label, s=58, edgecolor="white", linewidth=0.7)
    ax.set_title("AutoEncoder latent space colored by Node-Coded")
    ax.set_xlabel(f"Latent PC1 ({exp[0] * 100:.1f}%)")
    ax.set_ylabel(f"Latent PC2 ({exp[1] * 100:.1f}%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "latent_node_pca.png", dpi=300)
    plt.close(fig)

    base = predict_mlp(mlp, z_all, device)
    rng = np.random.default_rng(SEED)
    rows = []
    for i, feature in enumerate(state.selected_features):
        perturbed = x_all.copy()
        perturbed[:, i] = rng.permutation(perturbed[:, i])
        prob = predict_mlp(mlp, encode(dae, perturbed, device), device)
        rows.append({"feature": feature, "importance": float(np.mean(np.abs(prob - base)))})
    importance = pd.DataFrame(rows).join(metadata, on="feature").sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "top_protein_importance.csv", index=False, encoding="utf-8-sig")
    top = importance.head(15).iloc[::-1]
    labels = top["gene_symbol"].fillna("").replace("", np.nan).fillna(top["feature"])
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(labels, top["importance"], color="#54A24B")
    ax.set_xlabel("Mean absolute probability change")
    ax.set_title("Top protein perturbation importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_protein_importance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_df, y, clinical, metadata = load_data()
    print(f"Loaded matrix: {len(x_df)} samples x {x_df.shape[1]} protein features")
    print(f"Node-Coded counts: Negative={(y == 0).sum()}, Positive={(y == 1).sum()}")
    metrics_df, oof = run_cv(x_df, y, device)
    metrics_df.to_csv(RESULTS_DIR / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summarize(metrics_df)
    summary.to_csv(RESULTS_DIR / "cv_metrics_summary.csv", index=False, encoding="utf-8-sig")
    clinical.to_csv(RESULTS_DIR / "node_clinical_labels.csv", encoding="utf-8-sig")
    plot_outputs(summary, oof, clinical)
    rows = []
    for method in METHOD_NAMES:
        part = oof[oof["method"] == method]
        precision, recall, thresholds = precision_recall_curve(part["y_true"], part["y_prob"])
        for i in range(len(precision)):
            rows.append({"method": method, "precision": precision[i], "recall": recall[i], "threshold": thresholds[i] if i < len(thresholds) else np.nan})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "precision_recall_points.csv", index=False, encoding="utf-8-sig")
    train_final(x_df, y, metadata, device)
    print(summary[["method", "Balanced Accuracy_mean", "F1_mean", "ROC-AUC_mean", "PR-AUC_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
