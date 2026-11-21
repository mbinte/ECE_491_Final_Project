"""Train and evaluate a MobileNetV2 wildfire image classifier."""
import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from dataset import CLASSES, split_dataset, write_manifest

HERE = Path(__file__).resolve().parent
SEED = 491
RECALL_TARGET = 0.99
MODEL_NAME = "wildfire-mobilenet-v2.pt"


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label, *_ = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, label


def preprocessing():
    return MobileNet_V2_Weights.IMAGENET1K_V2.transforms(crop_size=128, resize_size=144)


def build_model(pretrained):
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Linear(model.last_channel, len(CLASSES))
    return model


@torch.inference_mode()
def extract(model, rows, transform, device):
    features, labels = [], []
    batches = DataLoader(Images(rows, transform), batch_size=64)
    model.eval()
    for x, y in batches:
        pooled = model.features(x.to(device)).mean(dim=(2, 3))
        features.append(pooled.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def logits_metrics(logits, y, threshold=0.5):
    prob = logits.softmax(dim=1)[:, 1].numpy()
    pred = torch.from_numpy(prob > threshold).long()
    conf = torch.bincount(2 * y + pred, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = conf.flatten().tolist()
    def ratio(num, den):
        return num / den if den else 0.0
    order = np.argsort(-prob, kind="stable")
    truth = y.numpy()[order]
    boundaries = np.r_[np.flatnonzero(np.diff(prob[order])), len(y) - 1]
    tpr = np.r_[0.0, truth.cumsum()[boundaries] / (tp + fn)]
    fpr = np.r_[0.0, (1 - truth).cumsum()[boundaries] / (tn + fp)]
    auc = ((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2).sum()
    return dict(loss=nn.functional.cross_entropy(logits, y).item(),
                accuracy=ratio(tn + tp, len(y)), precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                roc_auc=float(auc), confusion_matrix=conf.tolist()), fpr, tpr


def assess(validation_logits, validation_labels, test_logits, test_labels):
    """Select a fire threshold from validation positives before judging test scores."""
    with torch.no_grad():
        positives = validation_logits.softmax(dim=1)[validation_labels == 1, 1]
        rank = math.ceil(RECALL_TARGET * len(positives))
        cutoff = positives.sort(descending=True).values[rank - 1].numpy()
        # Use a strict > rule, consistent with argmax tie-breaking at 0.5.
        threshold = min(0.5, float(np.nextafter(cutoff, np.float32(-np.inf))))
    validation, _, _ = logits_metrics(validation_logits, validation_labels, threshold)
    baseline, _, _ = logits_metrics(test_logits, test_labels)
    test, fpr, tpr = logits_metrics(test_logits, test_labels, threshold)
    calibration = dict(target_validation_recall=RECALL_TARGET, threshold=threshold,
                       validation=validation, baseline_threshold=0.5, baseline_test=baseline)
    return calibration, test, fpr, tpr


def save_checkpoint(model, output, threshold):
    torch.save(dict(state_dict=model.cpu().state_dict(), threshold=threshold),
               output / MODEL_NAME)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(saved.get("state_dict", saved))
    return model.eval(), float(saved.get("threshold", 0.5))


def make_plots(output, history, test, fpr, tpr):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            ax.plot(range(1, len(history) + 1), [r[split][key] for r in history],
                    label=split.title())
        ax.set(xlabel="Epoch", ylabel=key.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.imshow(test["confusion_matrix"], cmap="Blues", alpha=0.65)
    for i, row in enumerate(test["confusion_matrix"]):
        for j, n in enumerate(row):
            ax.text(j, i, n, ha="center", va="center")
    ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Prediction", ylabel="True label", title="Test confusion matrix")
    ax = axes[1]
    ax.plot(fpr, tpr, label=f"AUC {test['roc_auc']:.3f}")
    tn, fp = test["confusion_matrix"][0]
    ax.scatter(fp / (tn + fp), test["recall"], label="Selected threshold", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "test-evaluation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--recordings", type=Path, help="Complete CSV with path and recording_id columns")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1:
        parser.error("--epochs and --threads must be positive")
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    splits, audit = split_dataset(args.data, SEED, args.recordings)
    write_manifest(args.output / "split.csv", args.data, splits)
    sizes = {name: {c: sum(r[1] == i for r in rows) for i, c in enumerate(CLASSES)}
             for name, rows in splits.items()}
    print("Data audit", audit, "split sizes", sizes, flush=True)

    transform = preprocessing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    features = {}
    for split in ("train", "validation"):
        x, y = extract(model, splits[split], transform, device)
        # Cloning outside inference mode lets autograd save the features.
        features[split] = (x.clone(), y.clone())
        print(f"Extracted {len(y)} {split} feature vectors", flush=True)
    head = model.classifier.cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    train_x, train_y = features["train"]
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        for indices in torch.randperm(len(train_y)).split(128):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(head(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            stats = {name: logits_metrics(head(x), y)[0] for name, (x, y) in features.items()}
        history.append(stats)
        val_loss = stats["validation"]["loss"]
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(head.state_dict())
        print(f"Epoch {epoch}: validation loss {val_loss:.4f}", flush=True)

    head.load_state_dict(best_state)
    test_x, test_y = extract(model, splits["test"], transform, device)
    with torch.no_grad():
        validation_x, validation_y = features["validation"]
        calibration, test, fpr, tpr = assess(
            head(validation_x), validation_y, head(test_x), test_y)
    save_checkpoint(model, args.output, calibration["threshold"])
    make_plots(args.output, history, test, fpr, tpr)
    metrics = dict(seed=SEED, audit=audit, counts=sizes, best_epoch=best_epoch,
                   test=test, calibration=calibration, history=history)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Held-out test", test, flush=True)


if __name__ == "__main__":
    main()
