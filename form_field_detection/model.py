"""Faster R-CNN model for form field detection.

Uses a pretrained ResNet backbone with Feature Pyramid Network (FPN) for
detecting form fields (text fields, checkboxes, radio buttons, buttons,
signatures, dropdowns, list boxes) in rendered PDF page images.
"""

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def build_model(num_classes=8, backbone_name="resnet50", pretrained=True):
    """Build a Faster R-CNN model for form field detection.

    Args:
        num_classes: Number of classes including background.
                     Default 8 = 7 field types + 1 background.
        backbone_name: "resnet50" or "resnet101".
        pretrained: Whether to use pretrained backbone weights.

    Returns:
        A Faster R-CNN model ready for training.
    """
    if backbone_name == "resnet50":
        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights
        )
    elif backbone_name == "resnet101":
        # Build with ResNet-101 backbone
        weights_backbone = "DEFAULT" if pretrained else None
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=weights_backbone,
        )
        # Form fields tend to be rectangular; use aspect ratios that
        # reflect typical field shapes.
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=anchor_ratios
        )
        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Replace the classifier head to match our number of classes
    if backbone_name == "resnet50":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )

    # Adjust anchor generator for form field shapes:
    # Form fields are often wider than tall (text inputs, dropdowns)
    # or roughly square (checkboxes, radio buttons).
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    anchor_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=anchor_ratios
    )

    return model
