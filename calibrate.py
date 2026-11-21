"""Recompute the validation-selected threshold for an existing checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import torch

from dataset import CLASSES
from train import (HERE, MODEL_NAME, assess, extract, load_checkpoint,
                   make_plots, preprocessing, save_checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.results
    model, _ = load_checkpoint(output / MODEL_NAME)
    metrics = json.loads((output / "metrics.json").read_text())
    with (output / "split.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    values = []
    with torch.inference_mode():
        for split in ("validation", "test"):
            images = [(args.data / row["path"], CLASSES.index(row["class"]))
                      for row in rows if row["split"] == split]
            x, y = extract(model, images, preprocessing(), torch.device("cpu"))
            values.extend((model.classifier(x), y))
    calibration, test, fpr, tpr = assess(*values)
    metrics.update(calibration=calibration, test=test)
    make_plots(output, metrics["history"], test, fpr, tpr)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_checkpoint(model, output, calibration["threshold"])
    print(f"Fire threshold: {calibration['threshold']:.6f}. Test recall: {test['recall']:.2%}.")


if __name__ == "__main__":
    main()
