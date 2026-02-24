"""PyTorch Dataset for form field detection.

Loads rendered PDF page images and their field annotations (bounding boxes
and class labels) for training and evaluation of the detection model.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class FormFieldDataset(Dataset):
    """Dataset of PDF page images with field bounding box annotations.

    Each sample returns:
        image: Tensor [3, H, W] normalized to [0, 1]
        target: dict with keys:
            - boxes: Tensor [N, 4] in (x_min, y_min, x_max, y_max) format
            - labels: Tensor [N] integer class labels
            - image_id: Tensor [1] unique image index
            - area: Tensor [N] box areas
            - iscrowd: Tensor [N] always 0 (no crowd annotations)
    """

    def __init__(self, images_dir, annotations_dir, file_list=None,
                 augment=False):
        """
        Args:
            images_dir: Path to directory containing page images.
            annotations_dir: Path to directory containing JSON annotations.
            file_list: Optional list of annotation filenames to use
                       (for train/val splitting). If None, uses all.
            augment: Whether to apply data augmentation during training.
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.augment = augment

        if file_list is not None:
            self.annotation_files = sorted(file_list)
        else:
            self.annotation_files = sorted(
                f.name for f in self.annotations_dir.glob("*.json")
            )

        # Filter to only include samples that have at least one field
        self.annotation_files = [
            f for f in self.annotation_files
            if self._has_fields(f)
        ]

    def _has_fields(self, annotation_file):
        path = self.annotations_dir / annotation_file
        with open(path) as f:
            data = json.load(f)
        return len(data.get("fields", [])) > 0

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        # Load annotation
        ann_path = self.annotations_dir / self.annotation_files[idx]
        with open(ann_path) as f:
            annotation = json.load(f)

        # Load image
        image_path = self.images_dir / annotation["image_file"]
        image = Image.open(image_path).convert("RGB")

        # Extract boxes and labels
        boxes = []
        labels = []
        for field in annotation["fields"]:
            bbox = field["bbox"]
            # Ensure valid box (x_max > x_min, y_max > y_min)
            x_min = min(bbox[0], bbox[2])
            y_min = min(bbox[1], bbox[3])
            x_max = max(bbox[0], bbox[2])
            y_max = max(bbox[1], bbox[3])
            if x_max - x_min < 1 or y_max - y_min < 1:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(field["label"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transforms
        if self.augment:
            image, target = self._augment(image, target)

        image = T.ToTensor()(image)

        return image, target

    def _augment(self, image, target):
        """Apply simple data augmentation.

        - Random horizontal flip
        - Random brightness/contrast jitter
        """
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = T.functional.hflip(image)
            boxes = target["boxes"]
            w = image.size[0] if hasattr(image, "size") else image.width
            # Flip x coordinates: new_x = width - old_x
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        # Color jitter (doesn't affect boxes)
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        image = jitter(image)

        return image, target


def build_datasets(processed_dir, train_split=0.8, augment_train=True):
    """Build train and validation datasets from processed data.

    Args:
        processed_dir: Directory containing images/ and annotations/ subdirs.
        train_split: Fraction of data to use for training.
        augment_train: Whether to augment the training set.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    processed_dir = Path(processed_dir)
    images_dir = processed_dir / "images"
    annotations_dir = processed_dir / "annotations"

    # Get all annotation files
    all_files = sorted(f.name for f in annotations_dir.glob("*.json"))
    np.random.seed(42)
    np.random.shuffle(all_files)

    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    train_dataset = FormFieldDataset(
        images_dir, annotations_dir, train_files, augment=augment_train
    )
    val_dataset = FormFieldDataset(
        images_dir, annotations_dir, val_files, augment=False
    )

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset


def collate_fn(batch):
    """Custom collate function for variable-sized targets."""
    return tuple(zip(*batch))
