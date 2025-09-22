"""Train image, chemistry, and fusion models for the cp-tox-fusion-mini project."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .extract_cp_features import CHANNEL_SUFFIXES, parse_metadata
from .utils import (
    configure_logging,
    data_dir,
    ensure_dir,
    find_image_triplets,
    reports_dir,
    safe_relative_path,
    save_json,
    set_seed,
)

LOGGER = logging.getLogger("cp_tox.training")
DEVICE = torch.device("cpu")


@dataclass
class ImageSample:
    """Metadata for an image tile used in CNN training."""

    paths: List[Path]
    plate: str
    compound_id: str
    well: str
    site: str
    label: float


class ChannelMixer(nn.Module):
    """Learned per-channel weights applied before convolutions."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weight, dim=0)
        return x * weights.view(1, -1, 1, 1)

    def importances(self) -> np.ndarray:
        weights = torch.softmax(self.weight.detach(), dim=0)
        return weights.cpu().numpy()


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128) -> None:
        super().__init__()
        self.mixer = ChannelMixer(in_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.relu(x)


class ImageClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding).squeeze(-1)
        return logits, embedding

    def channel_importances(self) -> np.ndarray:
        return self.encoder.mixer.importances()


class CellPaintingDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], image_size: int = 128) -> None:
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, ImageSample]:
        sample = self.samples[idx]
        stack = [self._load_channel(path) for path in sample.paths]
        array = np.stack(stack, axis=0)
        tensor = torch.from_numpy(array)
        if self.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor, torch.tensor(sample.label, dtype=torch.float32), sample

    @staticmethod
    def _load_channel(path: Path) -> np.ndarray:
        from skimage.io import imread

        channel = imread(path).astype(np.float32)
        if channel.max() > 0:
            channel /= channel.max()
        return channel


def collate_fn(batch):
    tensors, labels, metadata = zip(*batch)
    return torch.stack(tensors), torch.stack(labels), metadata


def build_image_samples(image_dir: Path, cp_df: pd.DataFrame) -> List[ImageSample]:
    label_lookup = (
        cp_df.set_index(["plate", "compound_id", "well"])["viability_label"].to_dict()
    )
    triplets = find_image_triplets(image_dir, CHANNEL_SUFFIXES)
    samples: List[ImageSample] = []
    for channels in triplets:
        meta = parse_metadata(channels[0].stem)
        key = (meta["plate"], meta["compound"], meta["well"])
        if key not in label_lookup:
            continue
        samples.append(
            ImageSample(
                paths=channels,
                plate=meta["plate"],
                compound_id=meta["compound"],
                well=meta["well"],
                site=meta["site"],
                label=float(label_lookup[key]),
            )
        )
    if not samples:
        raise RuntimeError("No image samples found with matching labels; check mapping")
    return samples


def compute_metrics(y_true: Iterable[float], y_scores: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true))
    y_scores_arr = np.asarray(list(y_scores))
    preds = (y_scores_arr >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, preds)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_scores_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["average_precision"] = float(average_precision_score(y_true_arr, y_scores_arr))
    except ValueError:
        metrics["average_precision"] = float("nan")
    return metrics


def plot_curves(y_true: np.ndarray, y_scores: np.ndarray, prefix: str, output_dir: Path) -> Dict[str, str]:
    ensure_dir(output_dir)
    roc_path = output_dir / f"{prefix}_roc.png"
    pr_path = output_dir / f"{prefix}_pr.png"

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()
    except ValueError:
        roc_path = None

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200)
        plt.close()
    except ValueError:
        pr_path = None

    return {
        "roc_curve": roc_path.name if roc_path else "",
        "pr_curve": pr_path.name if pr_path else "",
    }


def train_image_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
) -> Tuple[ImageClassifier, Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
    model = ImageClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_val_loss = math.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        val_loss = evaluate_loss(model, val_loader, criterion)
        LOGGER.info("Epoch %d | train %.4f | val %.4f", epoch + 1, avg_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    metrics = {}
    predictions = {}
    for split, loader in {"train": train_loader, "val": val_loader, "test": test_loader}.items():
        metrics[split] = evaluate_split(model, loader)
        predictions[split] = collect_predictions(model, loader, split)
    return model, metrics, predictions


def evaluate_loss(model: ImageClassifier, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            total += loss.item() * inputs.size(0)
            count += inputs.size(0)
    return total / max(count, 1)


def evaluate_split(model: ImageClassifier, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    scores: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            logits, _ = model(inputs.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(targets.numpy().tolist())
    return compute_metrics(labels, scores)


def collect_predictions(model: ImageClassifier, loader: DataLoader, split_name: str) -> pd.DataFrame:
    model.eval()
    records = []
    with torch.no_grad():
        for inputs, targets, metadata in loader:
            logits, embeddings = model(inputs.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            embeddings_np = embeddings.cpu().numpy()
            for prob, target, emb, meta in zip(probs, targets.numpy(), embeddings_np, metadata):
                record = {
                    "compound_id": meta.compound_id,
                    "well": meta.well,
                    "site": meta.site,
                    "true_label": float(target),
                    "probability": float(prob),
                    "split": split_name,
                }
                for idx, value in enumerate(emb):
                    record[f"img_emb_{idx:03d}"] = float(value)
                records.append(record)
    return pd.DataFrame.from_records(records)


def lightgbm_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[feature_cols],
        train_df[label_col],
        eval_set=[(val_df[feature_cols], val_df[label_col])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    metrics = {}
    predictions: Dict[str, pd.DataFrame] = {}
    for split_name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        if split_df.empty:
            metrics[split_name] = {"accuracy": float("nan"), "roc_auc": float("nan"), "average_precision": float("nan")}
            predictions[split_name] = pd.DataFrame()
            continue
        y_pred = model.predict_proba(split_df[feature_cols])[:, 1]
        metrics[split_name] = compute_metrics(split_df[label_col], y_pred)
        pred_df = split_df[["compound_id"]].copy()
        pred_df["probability"] = y_pred
        pred_df["true_label"] = split_df[label_col].to_numpy()
        pred_df["split"] = split_name
        predictions[split_name] = pred_df
    return model, metrics, predictions


def late_fusion_training(
    fusion_df: pd.DataFrame,
    splits: Dict[str, List[str]],
    embed_cols: List[str],
    chem_cols: List[str],
    label_col: str,
) -> Tuple[LogisticRegression, Dict[str, Dict[str, float]], Dict[str, float], Dict[str, pd.DataFrame]]:
    scaler = StandardScaler()
    pca = PCA(n_components=min(64, len(chem_cols)))
    fusion_df = fusion_df.dropna(subset=embed_cols + chem_cols + [label_col])

    chem_scaled = scaler.fit_transform(fusion_df[chem_cols])
    chem_pca = pca.fit_transform(chem_scaled)

    embeddings = fusion_df[embed_cols].to_numpy()
    X = np.concatenate([embeddings, chem_pca], axis=1)
    y = fusion_df[label_col].to_numpy()

    split_masks = {
        name: fusion_df["compound_id"].isin(compounds)
        for name, compounds in splits.items()
    }

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X[split_masks["train"]], y[split_masks["train"]])

    metrics = {}
    predictions: Dict[str, pd.DataFrame] = {}
    for name, mask in split_masks.items():
        if mask.sum() == 0:
            metrics[name] = {"accuracy": float("nan"), "roc_auc": float("nan"), "average_precision": float("nan")}
            predictions[name] = pd.DataFrame()
            continue
        scores = model.predict_proba(X[mask])[:, 1]
        metrics[name] = compute_metrics(y[mask], scores)
        pred_df = fusion_df.loc[mask, ["compound_id", "chem_compound_id"]].copy()
        pred_df["probability"] = scores
        pred_df["true_label"] = y[mask]
        pred_df["split"] = name
        predictions[name] = pred_df
    channel_weights = {"pca_explained_variance": float(pca.explained_variance_ratio_.sum())}
    return model, metrics, channel_weights, predictions


def subset_by_split(df: pd.DataFrame, splits: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["compound_id"].isin(splits["train"])]
    val_df = df[df["compound_id"].isin(splits["val"])]
    test_df = df[df["compound_id"].isin(splits["test"])]
    return train_df, val_df, test_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=data_dir("processed"))
    parser.add_argument("--raw_image_dir", type=Path, default=data_dir("raw", "bbbc021"))
    parser.add_argument("--splits", type=Path, default=data_dir("processed", "splits.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=reports_dir())
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    set_seed(args.seed)

    processed_dir = args.processed_dir
    splits = json.loads(args.splits.read_text())

    cp_dataset = pd.read_parquet(processed_dir / "cp_dataset.parquet")
    chem_dataset = pd.read_parquet(processed_dir / "chem_dataset.parquet")
    fusion_path = processed_dir / "fusion_dataset.parquet"
    fusion_dataset = pd.read_parquet(fusion_path) if fusion_path.exists() else pd.DataFrame()

    image_samples = build_image_samples(args.raw_image_dir, cp_dataset)
    sample_splits = {
        name: [sample for sample in image_samples if sample.compound_id in set(compounds)]
        for name, compounds in splits["cp"].items()
    }
    datasets = {
        name: CellPaintingDataset(sorted(samples, key=lambda s: (s.plate, s.well, s.site)))
        for name, samples in sample_splits.items()
    }
    dataloaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=(name == "train"), collate_fn=collate_fn)
        for name, dataset in datasets.items()
    }

    image_model, image_metrics, image_predictions = train_image_model(
        dataloaders["train"], dataloaders["val"], dataloaders["test"], epochs=args.epochs
    )

    image_predictions_df = pd.concat(image_predictions.values(), ignore_index=True)
    embedding_cols = [col for col in image_predictions_df.columns if col.startswith("img_emb_")]
    embedding_df = image_predictions_df[
        ["compound_id", "well", "site"] + embedding_cols + ["probability", "true_label", "split"]
    ]
    embedding_df.to_parquet(processed_dir / "image_embeddings.parquet", index=False)

    chem_feature_cols = [col for col in chem_dataset.columns if col.startswith("chem_") and col != "chem_compound_id"]
    chem_label_col = "toxicity_label"
    chem_train, chem_val, chem_test = subset_by_split(chem_dataset, splits["chem"])
    chem_model, chem_metrics, chem_predictions = lightgbm_train(
        chem_train, chem_val, chem_test, chem_feature_cols, chem_label_col
    )

    fusion_metrics = {}
    fusion_extras = {}
    fusion_predictions: Dict[str, pd.DataFrame] = {}
    if not fusion_dataset.empty:
        embed_cols = [col for col in embedding_df.columns if col.startswith("img_emb_")]
        agg_embeddings = embedding_df.groupby("compound_id")[embed_cols].mean().reset_index()
        fusion_dataset = fusion_dataset.merge(agg_embeddings, on="compound_id", how="inner")
        chem_cols = [
            col
            for col in fusion_dataset.columns
            if col.startswith("chem_") and col not in {"chem_compound_id"}
        ]
        embed_cols = [col for col in fusion_dataset.columns if col.startswith("img_emb_")]
        fusion_model, fusion_metrics, fusion_extras, fusion_predictions = late_fusion_training(
            fusion_dataset,
            splits["fusion"],
            embed_cols,
            chem_cols,
            label_col="toxicity_label",
        )
    else:
        LOGGER.warning("Skipping fusion model training; no fused dataset found")

    metrics_summary = {
        "image": image_metrics,
        "chem": chem_metrics,
        "fusion": fusion_metrics,
    }

    ensure_dir(args.output_dir)
    save_json(metrics_summary, args.output_dir / "model_metrics.json")

    channel_weights = image_model.channel_importances().tolist()
    channel_info = {"channel_importances": channel_weights, "channel_order": ["DNA", "Actin", "Tubulin"]}
    save_json(channel_info, args.output_dir / "channel_weights.json")

    test_split = dataloaders["test"]
    model = image_model.eval()
    test_scores = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels, _ in test_split:
            logits, _ = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_scores.extend(probs.tolist())
            test_labels.extend(labels.numpy().tolist())
    plot_curves(np.array(test_labels), np.array(test_scores), "image", args.output_dir)

    if not chem_test.empty:
        chem_scores = chem_model.predict_proba(chem_test[chem_feature_cols])[:, 1]
        plot_curves(chem_test[chem_label_col].to_numpy(), chem_scores, "chem", args.output_dir)

    if fusion_metrics:
        fusion_mask = fusion_dataset["compound_id"].isin(splits["fusion"]["test"])
        if fusion_mask.any():
            embed_cols = [col for col in fusion_dataset.columns if col.startswith("img_emb_")]
            chem_cols = [
                col
                for col in fusion_dataset.columns
                if col.startswith("chem_") and col not in {"chem_compound_id"}
            ]
            scaler = StandardScaler()
            pca = PCA(n_components=min(64, len(chem_cols)))
            chem_scaled = scaler.fit_transform(fusion_dataset[chem_cols])
            chem_pca = pca.fit_transform(chem_scaled)
            embeddings = fusion_dataset[embed_cols].to_numpy()
            X = np.concatenate([embeddings, chem_pca], axis=1)
            y = fusion_dataset["toxicity_label"].to_numpy()
            scores = fusion_model.predict_proba(X[fusion_mask])[:, 1]
            plot_curves(y[fusion_mask], scores, "fusion", args.output_dir)

    if fusion_extras:
        save_json(fusion_extras, args.output_dir / "fusion_metadata.json")

    predictions_dir = processed_dir / "predictions"
    ensure_dir(predictions_dir)
    image_predictions_df.to_parquet(predictions_dir / "image_predictions.parquet", index=False)
    for split, pred_df in chem_predictions.items():
        if not pred_df.empty:
            pred_df.to_parquet(predictions_dir / f"chem_predictions_{split}.parquet", index=False)
    if fusion_predictions:
        for split, pred_df in fusion_predictions.items():
            if not pred_df.empty:
                pred_df.to_parquet(predictions_dir / f"fusion_predictions_{split}.parquet", index=False)

    LOGGER.info("Training complete; metrics stored in %s", safe_relative_path(args.output_dir / "model_metrics.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
