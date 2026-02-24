# HiPhiLLM - Form Field Detection Model

Trains on interactive PDFs with AcroFields to learn field positions, then detects those field positions on flat (non-interactive) PDF forms.

## How It Works

1. **Extract**: Parse interactive PDFs to get ground truth bounding boxes and field types (text fields, checkboxes, radio buttons, buttons, signatures, dropdowns, list boxes) from AcroField metadata
2. **Render**: Convert each PDF page to an image, mapping field coordinates to pixel space
3. **Train**: Train a Faster R-CNN object detection model on the (image, bounding_boxes) pairs
4. **Predict**: Run the trained model on flat PDFs to detect field positions without AcroField metadata

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

### Step 1: Prepare Training Data

Place interactive PDFs (forms with fillable AcroFields) in `data/acro_pdfs/`, then extract the field annotations:

```bash
python run.py extract
```

This renders each page as an image and saves the field bounding boxes as JSON annotations under `data/processed/`.

### Step 2: Check Dataset

View statistics about the extracted dataset:

```bash
python run.py stats
```

### Step 3: Train the Model

```bash
python run.py train
```

Training configuration is in `config.yaml`. Checkpoints are saved to `checkpoints/`.

### Step 4: Run Inference on Flat PDFs

Place flat (non-interactive) PDFs in `data/flat_pdfs/`, then run:

```bash
python run.py predict
```

Or target a specific PDF:

```bash
python run.py predict --pdf path/to/form.pdf
```

Outputs are saved to `output/` as JSON predictions and annotated images.

### Step 5: Evaluate

Evaluate the model against annotated data:

```bash
python run.py evaluate --checkpoint checkpoints/best_model.pth
```

## Project Structure

```
form_field_detection/
    __init__.py          - Package init
    extract_fields.py    - AcroField extraction from interactive PDFs
    dataset.py           - PyTorch Dataset for training
    model.py             - Faster R-CNN model definition
    train.py             - Training loop
    predict.py           - Inference on flat PDFs
    evaluate.py          - mAP evaluation
    visualize.py         - Visualization and dataset statistics
config.yaml              - All configuration (data, model, training, inference)
run.py                   - Main entry point
```

## Configuration

Edit `config.yaml` to adjust:

- **data**: PDF directories, render DPI, train/val split
- **model**: Backbone (resnet50/resnet101), number of field classes, pretrained weights
- **training**: Batch size, learning rate, epochs, checkpointing
- **inference**: Confidence threshold, NMS threshold, output directory

## Field Types Detected

| Class | Label |
|-------|-------|
| text_field | 1 |
| checkbox | 2 |
| radio_button | 3 |
| button | 4 |
| signature | 5 |
| dropdown | 6 |
| list_box | 7 |

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Detection head**: Faster R-CNN
- **Anchors**: Tuned for form field shapes (aspect ratios 0.25 to 4.0, sizes 16 to 256 pixels)
- **Transfer learning**: Pretrained on COCO, fine-tuned on form fields
