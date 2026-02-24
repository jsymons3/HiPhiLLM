"""Main entry point for the form field detection pipeline.

Usage:
    python run.py extract   - Extract fields from interactive PDFs
    python run.py train     - Train the detection model
    python run.py predict   - Run inference on flat PDFs
    python run.py evaluate  - Evaluate model on annotated data
    python run.py stats     - Print dataset statistics
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Form Field Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps to use:
  1. Place interactive PDFs (with AcroFields) in data/acro_pdfs/
  2. Run: python run.py extract
  3. Run: python run.py train
  4. Place flat PDFs in data/flat_pdfs/
  5. Run: python run.py predict
        """,
    )
    parser.add_argument(
        "command",
        choices=["extract", "train", "predict", "evaluate", "stats"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (for predict/evaluate)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Single PDF file path (for predict)",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="PDF directory (for predict)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override render DPI from config",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.command == "extract":
        from form_field_detection.extract_fields import process_pdf_directory
        dpi = args.dpi or config["data"]["render_dpi"]
        process_pdf_directory(
            config["data"]["acro_pdf_dir"],
            config["data"]["processed_dir"],
            render_dpi=dpi,
        )

    elif args.command == "train":
        from form_field_detection.train import train
        train(args.config)

    elif args.command == "predict":
        from form_field_detection.predict import run_inference
        run_inference(
            args.config,
            checkpoint_path=args.checkpoint,
            pdf_path=args.pdf,
            pdf_dir=args.pdf_dir,
        )

    elif args.command == "evaluate":
        from form_field_detection.evaluate import evaluate_on_dataset
        checkpoint = args.checkpoint or "checkpoints/best_model.pth"
        evaluate_on_dataset(
            checkpoint,
            config["data"]["processed_dir"],
        )

    elif args.command == "stats":
        from form_field_detection.visualize import dataset_statistics
        dataset_statistics(config["data"]["processed_dir"])


if __name__ == "__main__":
    main()
