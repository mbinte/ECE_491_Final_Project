"""Predict wildfire presence in one or more local images."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps
import torch

from dataset import CLASSES
from train import HERE, MODEL_NAME, load_checkpoint, preprocessing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=HERE / "results" / MODEL_NAME)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, threshold = load_checkpoint(args.model)
    transform = preprocessing()
    with torch.inference_mode():
        for path in args.images:
            with Image.open(path) as image:
                tensor = transform(ImageOps.exif_transpose(image).convert("RGB"))
            probabilities = model(tensor.unsqueeze(0)).softmax(dim=1)[0]
            predicted = CLASSES[int(probabilities[1].item() > threshold)]
            print(f"{path}: {predicted} (fire score {probabilities[1].item():.4f})")


if __name__ == "__main__":
    main()
