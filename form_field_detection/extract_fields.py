"""Extract AcroField positions from interactive PDFs.

Reads PDF forms with AcroFields (text fields, checkboxes, radio buttons,
buttons, signatures, dropdowns, list boxes) and extracts their bounding
box coordinates and field types. Renders each page as an image to create
training pairs: (page_image, field_annotations).
"""

import json
import os
from pathlib import Path

import fitz  # PyMuPDF


# Mapping from PDF widget type integers to human-readable class names.
# fitz.PDF_WIDGET_TYPE_* constants:
#   0 = unknown, 1 = button, 2 = checkbox, 3 = combobox,
#   4 = listbox, 5 = radiobutton, 6 = signature, 7 = text
WIDGET_TYPE_MAP = {
    0: "unknown",
    1: "button",
    2: "checkbox",
    3: "dropdown",
    4: "list_box",
    5: "radio_button",
    6: "signature",
    7: "text_field",
}

# Class name to integer label (for the detection model).
# 0 is reserved for background in Faster R-CNN.
CLASS_TO_LABEL = {
    "text_field": 1,
    "checkbox": 2,
    "radio_button": 3,
    "button": 4,
    "signature": 5,
    "dropdown": 6,
    "list_box": 7,
}


def extract_fields_from_pdf(pdf_path, render_dpi=150):
    """Extract field annotations and rendered page images from a PDF.

    Args:
        pdf_path: Path to an interactive PDF with AcroFields.
        render_dpi: Resolution for rendering pages to images.

    Returns:
        List of dicts, one per page, each containing:
            - "page_num": int
            - "width": page width in pixels at render_dpi
            - "height": page height in pixels at render_dpi
            - "image": PIL-compatible pixmap
            - "fields": list of field dicts with keys:
                - "type": str (class name)
                - "label": int (class label)
                - "bbox": [x_min, y_min, x_max, y_max] in pixel coords
                - "name": str (field name from the PDF)
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        # Render page to image
        zoom = render_dpi / 72.0  # PDF default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)

        fields = []
        for widget in page.widgets():
            widget_type = widget.field_type
            type_name = WIDGET_TYPE_MAP.get(widget_type, "unknown")

            if type_name == "unknown":
                continue

            label = CLASS_TO_LABEL.get(type_name)
            if label is None:
                continue

            # Widget rect is in PDF points; scale to pixel coordinates
            rect = widget.rect
            bbox = [
                rect.x0 * zoom,
                rect.y0 * zoom,
                rect.x1 * zoom,
                rect.y1 * zoom,
            ]

            # Skip degenerate boxes
            if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                continue

            fields.append({
                "type": type_name,
                "label": label,
                "bbox": bbox,
                "name": widget.field_name or "",
            })

        pages_data.append({
            "page_num": page_num,
            "width": pixmap.width,
            "height": pixmap.height,
            "pixmap": pixmap,
            "fields": fields,
        })

    doc.close()
    return pages_data


def process_pdf_directory(pdf_dir, output_dir, render_dpi=150):
    """Process all PDFs in a directory and save images + annotations.

    Creates the following structure under output_dir:
        images/
            <pdf_stem>_page<N>.png
        annotations/
            <pdf_stem>_page<N>.json

    Args:
        pdf_dir: Directory containing interactive PDFs.
        output_dir: Directory to write processed data.
        render_dpi: DPI for rendering.

    Returns:
        List of annotation file paths created.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return annotation_files

    total_fields = 0
    total_pages = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            pages_data = extract_fields_from_pdf(str(pdf_path), render_dpi)
        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {e}")
            continue

        stem = pdf_path.stem

        for page_data in pages_data:
            page_num = page_data["page_num"]
            base_name = f"{stem}_page{page_num:03d}"

            # Save image
            image_path = images_dir / f"{base_name}.png"
            page_data["pixmap"].save(str(image_path))

            # Save annotation
            annotation = {
                "image_file": f"{base_name}.png",
                "width": page_data["width"],
                "height": page_data["height"],
                "fields": page_data["fields"],
            }
            annotation_path = annotations_dir / f"{base_name}.json"
            with open(annotation_path, "w") as f:
                json.dump(annotation, f, indent=2)

            annotation_files.append(str(annotation_path))
            total_fields += len(page_data["fields"])
            total_pages += 1

            field_count = len(page_data["fields"])
            if field_count > 0:
                print(f"  Page {page_num}: {field_count} fields extracted")

    print(f"\nExtraction complete:")
    print(f"  Pages processed: {total_pages}")
    print(f"  Total fields extracted: {total_fields}")
    print(f"  Output directory: {output_dir}")

    return annotation_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract AcroField positions from interactive PDFs"
    )
    parser.add_argument(
        "pdf_dir",
        help="Directory containing interactive PDFs with AcroFields",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for images and annotations",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for rendering PDF pages (default: 150)",
    )
    args = parser.parse_args()

    process_pdf_directory(args.pdf_dir, args.output_dir, args.dpi)
