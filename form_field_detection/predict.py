"""Inference script for detecting fields on flat (non-interactive) PDFs.

Loads a trained model and runs it on rendered PDF pages to predict
field bounding boxes and types. Outputs predictions as JSON and
optionally as annotated images.
"""

import json
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from .extract_fields import CLASS_TO_LABEL
from .model import build_model

# Reverse mapping: label -> class name
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# Colors for visualization (one per class)
CLASS_COLORS = {
    "text_field": "#2196F3",     # blue
    "checkbox": "#4CAF50",       # green
    "radio_button": "#FF9800",   # orange
    "button": "#F44336",         # red
    "signature": "#9C27B0",      # purple
    "dropdown": "#00BCD4",       # cyan
    "list_box": "#795548",       # brown
}


def load_model(checkpoint_path, device=None):
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: torch device. Auto-selects if None.

    Returns:
        (model, device) tuple with model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    num_classes = checkpoint.get("num_classes", 8)
    backbone = checkpoint.get("backbone", "resnet50")

    model = build_model(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict_page_image(model, image, device, confidence_threshold=0.5,
                       nms_threshold=0.3):
    """Run prediction on a single page image.

    Args:
        model: Trained Faster R-CNN model in eval mode.
        image: PIL Image of the page.
        device: torch device.
        confidence_threshold: Minimum score to keep a detection.
        nms_threshold: IoU threshold for NMS (applied by the model).

    Returns:
        List of prediction dicts with keys:
            - "type": str class name
            - "confidence": float score
            - "bbox": [x_min, y_min, x_max, y_max] in pixel coords
    """
    transform = T.ToTensor()
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    output = outputs[0]
    predictions = []

    for box, label, score in zip(
        output["boxes"], output["labels"], output["scores"]
    ):
        if score.item() < confidence_threshold:
            continue

        class_name = LABEL_TO_CLASS.get(label.item(), "unknown")
        if class_name == "unknown":
            continue

        predictions.append({
            "type": class_name,
            "confidence": round(score.item(), 4),
            "bbox": [round(c, 1) for c in box.tolist()],
        })

    return predictions


def predict_pdf(model, pdf_path, device, render_dpi=150,
                confidence_threshold=0.5, nms_threshold=0.3):
    """Run field detection on all pages of a flat PDF.

    Args:
        model: Trained model in eval mode.
        pdf_path: Path to the flat PDF.
        device: torch device.
        render_dpi: DPI for rendering pages.
        confidence_threshold: Minimum detection confidence.
        nms_threshold: NMS IoU threshold.

    Returns:
        List of page result dicts with keys:
            - "page_num": int
            - "width": int pixel width
            - "height": int pixel height
            - "predictions": list of prediction dicts
    """
    doc = fitz.open(pdf_path)
    zoom = render_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    results = []

    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=matrix)
        # Convert to PIL Image
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height],
                                pixmap.samples)

        preds = predict_page_image(
            model, image, device, confidence_threshold, nms_threshold
        )

        results.append({
            "page_num": page_num,
            "width": pixmap.width,
            "height": pixmap.height,
            "predictions": preds,
        })

    doc.close()
    return results


def draw_predictions(image, predictions):
    """Draw predicted bounding boxes on an image.

    Args:
        image: PIL Image.
        predictions: List of prediction dicts from predict_page_image.

    Returns:
        Annotated PIL Image.
    """
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        bbox = pred["bbox"]
        class_name = pred["type"]
        score = pred["confidence"]
        color = CLASS_COLORS.get(class_name, "#FFFFFF")

        # Draw box
        draw.rectangle(bbox, outline=color, width=2)

        # Draw label
        label_text = f"{class_name} {score:.2f}"
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label_text)
        text_bg = [text_bbox[0] - 1, text_bbox[1] - 1,
                   text_bbox[2] + 1, text_bbox[3] + 1]
        draw.rectangle(text_bg, fill=color)
        draw.text((bbox[0], bbox[1]), label_text, fill="white")

    return image


def run_inference(config_path="config.yaml", checkpoint_path=None,
                  pdf_path=None, pdf_dir=None):
    """Run inference on flat PDFs and save results.

    Args:
        config_path: Path to config file.
        checkpoint_path: Path to model checkpoint. If None, uses
                         checkpoints/best_model.pth.
        pdf_path: Path to a single PDF file.
        pdf_dir: Path to a directory of PDFs. Ignored if pdf_path is set.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    inf_cfg = config["inference"]
    data_cfg = config["data"]

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config["training"]["checkpoint_dir"], "best_model.pth"
        )

    output_dir = Path(inf_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, device = load_model(checkpoint_path)

    # Gather PDF files
    if pdf_path:
        pdf_files = [Path(pdf_path)]
    elif pdf_dir:
        pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    else:
        pdf_files = sorted(Path(data_cfg["flat_pdf_dir"]).glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found for inference.")
        return

    print(f"Running inference on {len(pdf_files)} PDF(s)...")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        results = predict_pdf(
            model, str(pdf_file), device,
            render_dpi=data_cfg["render_dpi"],
            confidence_threshold=inf_cfg["confidence_threshold"],
            nms_threshold=inf_cfg["nms_threshold"],
        )

        # Save JSON results
        json_path = output_dir / f"{pdf_file.stem}_predictions.json"
        with open(json_path, "w") as f:
            json.dump({
                "source_pdf": pdf_file.name,
                "render_dpi": data_cfg["render_dpi"],
                "confidence_threshold": inf_cfg["confidence_threshold"],
                "pages": results,
            }, f, indent=2)
        print(f"  Predictions saved to {json_path}")

        # Save annotated images
        doc = fitz.open(str(pdf_file))
        zoom = data_cfg["render_dpi"] / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_result in results:
            page_num = page_result["page_num"]
            preds = page_result["predictions"]
            if not preds:
                continue

            pixmap = doc[page_num].get_pixmap(matrix=matrix)
            image = Image.frombytes(
                "RGB", [pixmap.width, pixmap.height], pixmap.samples
            )
            annotated = draw_predictions(image, preds)

            img_path = output_dir / f"{pdf_file.stem}_page{page_num:03d}.png"
            annotated.save(str(img_path))
            print(f"  Page {page_num}: {len(preds)} fields detected "
                  f"-> {img_path}")

        doc.close()

    print(f"\nInference complete. Results in {output_dir}/")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Detect form fields on flat PDFs"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to a single PDF file",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directory of PDF files",
    )
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.pdf, args.pdf_dir)
