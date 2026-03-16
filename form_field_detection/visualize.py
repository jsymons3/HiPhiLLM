"""Visualization utilities for form field detection.

Provides functions to visualize ground truth annotations and predictions
side by side, and to generate summary statistics about the dataset.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .extract_fields import CLASS_TO_LABEL
from .predict import CLASS_COLORS


def visualize_annotation(image_path, annotation_path, output_path=None):
    """Draw ground truth boxes on an image.

    Args:
        image_path: Path to the page image.
        annotation_path: Path to the JSON annotation.
        output_path: Path to save the visualization. If None, displays it.
    """
    image = Image.open(image_path)
    with open(annotation_path) as f:
        annotation = json.load(f)

    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.imshow(image)

    for field in annotation["fields"]:
        bbox = field["bbox"]
        class_name = field["type"]
        color = CLASS_COLORS.get(class_name, "#FFFFFF")

        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 2, class_name,
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.set_title(f"Ground Truth: {Path(image_path).name}")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_comparison(image_path, annotation_path, predictions,
                         output_path=None):
    """Show ground truth and predictions side by side.

    Args:
        image_path: Path to the page image.
        annotation_path: Path to the JSON annotation.
        predictions: List of prediction dicts from predict_page_image.
        output_path: Path to save. If None, displays.
    """
    image = Image.open(image_path)
    with open(annotation_path) as f:
        annotation = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))

    # Ground truth
    ax1.imshow(image)
    ax1.set_title("Ground Truth", fontsize=14)
    for field in annotation["fields"]:
        bbox = field["bbox"]
        color = CLASS_COLORS.get(field["type"], "#FFFFFF")
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax1.add_patch(rect)
        ax1.text(bbox[0], bbox[1] - 2, field["type"], fontsize=6,
                 color="white",
                 bbox=dict(facecolor=color, alpha=0.8, pad=1))
    ax1.axis("off")

    # Predictions
    ax2.imshow(image)
    ax2.set_title("Predictions", fontsize=14)
    for pred in predictions:
        bbox = pred["bbox"]
        color = CLASS_COLORS.get(pred["type"], "#FFFFFF")
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax2.add_patch(rect)
        label = f"{pred['type']} {pred['confidence']:.2f}"
        ax2.text(bbox[0], bbox[1] - 2, label, fontsize=6,
                 color="white",
                 bbox=dict(facecolor=color, alpha=0.8, pad=1))
    ax2.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def dataset_statistics(processed_dir):
    """Print summary statistics about the extracted dataset.

    Args:
        processed_dir: Path to the processed data directory.
    """
    processed_dir = Path(processed_dir)
    annotations_dir = processed_dir / "annotations"

    ann_files = sorted(annotations_dir.glob("*.json"))
    if not ann_files:
        print("No annotation files found.")
        return

    total_fields = 0
    type_counts = Counter()
    fields_per_page = []
    box_widths = []
    box_heights = []

    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)
        fields = ann.get("fields", [])
        fields_per_page.append(len(fields))
        total_fields += len(fields)

        for field in fields:
            type_counts[field["type"]] += 1
            bbox = field["bbox"]
            box_widths.append(bbox[2] - bbox[0])
            box_heights.append(bbox[3] - bbox[1])

    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total pages:           {len(ann_files)}")
    print(f"Total fields:          {total_fields}")
    print(f"Avg fields per page:   {np.mean(fields_per_page):.1f}")
    print(f"Max fields per page:   {max(fields_per_page)}")
    print(f"Pages with no fields:  "
          f"{sum(1 for f in fields_per_page if f == 0)}")
    print()
    print("Field type distribution:")
    for field_type, count in type_counts.most_common():
        pct = 100 * count / total_fields if total_fields > 0 else 0
        print(f"  {field_type:15s}  {count:5d}  ({pct:5.1f}%)")
    print()
    if box_widths:
        print("Bounding box sizes (pixels):")
        print(f"  Width  - mean: {np.mean(box_widths):6.1f}, "
              f"median: {np.median(box_widths):6.1f}, "
              f"min: {min(box_widths):6.1f}, max: {max(box_widths):6.1f}")
        print(f"  Height - mean: {np.mean(box_heights):6.1f}, "
              f"median: {np.median(box_heights):6.1f}, "
              f"min: {min(box_heights):6.1f}, max: {max(box_heights):6.1f}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset visualization tools")
    parser.add_argument(
        "processed_dir",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics",
    )
    args = parser.parse_args()

    if args.stats:
        dataset_statistics(args.processed_dir)
