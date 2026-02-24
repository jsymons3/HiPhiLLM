"""Evaluation utilities for the form field detection model.

Computes per-class precision, recall, and mean Average Precision (mAP)
using IoU-based matching between predicted and ground truth boxes.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .extract_fields import CLASS_TO_LABEL
from .predict import LABEL_TO_CLASS, load_model


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x_min, y_min, x_max, y_max]."""
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return intersection / union


def compute_ap(precisions, recalls):
    """Compute Average Precision using the 11-point interpolation method."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_above:
            ap += max(precisions_above) / 11.0
    return ap


def evaluate_predictions(predictions, ground_truths, iou_threshold=0.5):
    """Compute per-class AP and overall mAP.

    Args:
        predictions: List of dicts, each with:
            - "bbox": [x_min, y_min, x_max, y_max]
            - "type": str class name
            - "confidence": float
            - "image_id": str or int
        ground_truths: List of dicts, each with:
            - "bbox": [x_min, y_min, x_max, y_max]
            - "type": str class name
            - "image_id": str or int
        iou_threshold: IoU threshold for considering a match.

    Returns:
        Dict with per-class AP and overall mAP.
    """
    # Group by class
    pred_by_class = defaultdict(list)
    gt_by_class = defaultdict(lambda: defaultdict(list))

    for pred in predictions:
        pred_by_class[pred["type"]].append(pred)

    for gt in ground_truths:
        gt_by_class[gt["type"]][gt["image_id"]].append(gt)

    results = {}
    all_aps = []

    for class_name in CLASS_TO_LABEL:
        class_preds = sorted(
            pred_by_class.get(class_name, []),
            key=lambda x: -x["confidence"],
        )
        class_gts = gt_by_class.get(class_name, {})

        total_gt = sum(len(gts) for gts in class_gts.values())
        if total_gt == 0:
            continue

        # Track which GT boxes have been matched
        matched = defaultdict(set)
        tp_list = []
        fp_list = []

        for pred in class_preds:
            img_id = pred["image_id"]
            gts_for_image = class_gts.get(img_id, [])

            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts_for_image):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched[img_id]:
                tp_list.append(1)
                fp_list.append(0)
                matched[img_id].add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision/recall curve
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / total_gt

        ap = compute_ap(precisions.tolist(), recalls.tolist())
        all_aps.append(ap)

        results[class_name] = {
            "AP": round(ap, 4),
            "total_predictions": len(class_preds),
            "total_ground_truths": total_gt,
            "true_positives": int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        }

    results["mAP"] = round(np.mean(all_aps), 4) if all_aps else 0.0
    return results


def evaluate_on_dataset(checkpoint_path, processed_dir, render_dpi=150,
                        confidence_threshold=0.3):
    """Evaluate a trained model on the validation set.

    Args:
        checkpoint_path: Path to model checkpoint.
        processed_dir: Path to processed data directory.
        render_dpi: DPI used during data extraction.
        confidence_threshold: Minimum confidence for predictions.

    Returns:
        Evaluation results dict.
    """
    processed_dir = Path(processed_dir)
    images_dir = processed_dir / "images"
    annotations_dir = processed_dir / "annotations"

    model, device = load_model(checkpoint_path)
    transform = T.ToTensor()

    # Use all annotations for evaluation
    all_predictions = []
    all_ground_truths = []

    ann_files = sorted(annotations_dir.glob("*.json"))
    print(f"Evaluating on {len(ann_files)} samples...")

    for ann_file in ann_files:
        with open(ann_file) as f:
            annotation = json.load(f)

        image_path = images_dir / annotation["image_file"]
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        image_id = ann_file.stem

        # Ground truths
        for field in annotation["fields"]:
            all_ground_truths.append({
                "bbox": field["bbox"],
                "type": field["type"],
                "image_id": image_id,
            })

        # Predictions
        img_tensor = transform(image).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])

        output = outputs[0]
        for box, label, score in zip(
            output["boxes"], output["labels"], output["scores"]
        ):
            if score.item() < confidence_threshold:
                continue
            class_name = LABEL_TO_CLASS.get(label.item())
            if class_name is None:
                continue
            all_predictions.append({
                "bbox": box.tolist(),
                "type": class_name,
                "confidence": score.item(),
                "image_id": image_id,
            })

    results = evaluate_predictions(all_predictions, all_ground_truths)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for class_name in CLASS_TO_LABEL:
        if class_name in results:
            r = results[class_name]
            print(f"  {class_name:15s}  AP={r['AP']:.4f}  "
                  f"TP={r['true_positives']}/{r['total_ground_truths']}  "
                  f"Preds={r['total_predictions']}")
    print(f"\n  {'mAP':15s}  {results['mAP']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate form field detector")
    parser.add_argument(
        "checkpoint",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    args = parser.parse_args()

    evaluate_on_dataset(args.checkpoint, args.data_dir, confidence_threshold=args.confidence)
