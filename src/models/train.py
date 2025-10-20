"""
YOLOv8 training script for drone detection.
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    name: str = "drone_detection",
    device: str = "",
    **kwargs
):
    """
    Train YOLOv8 model on drone detection dataset.

    Args:
        data_yaml: Path to dataset YAML file
        model: Pretrained model to use (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
        name: Experiment name
        device: Device to train on (auto-detect if empty)
        **kwargs: Additional training arguments
    """
    print(f"\n{'='*60}")
    print(f"Training YOLOv8 Model: {name}")
    print(f"{'='*60}\n")

    print(f"Dataset: {data_yaml}")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device if device else 'auto-detect'}\n")

    # Load model
    yolo_model = YOLO(model)

    # Train
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name,
        device=device,
        project="models/checkpoints",
        **kwargs
    )

    # Get best model path
    best_model = Path(f"models/checkpoints/{name}/weights/best.pt")
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best model saved to: {best_model}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for drone detection")

    # Dataset
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/drone-vs-bird-binary/data.yaml",
        help="Path to dataset YAML file"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model (yolov8n/s/m/l/x.pt)"
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="drone_detection", help="Experiment name")
    parser.add_argument("--device", type=str, default="", help="Device (empty for auto)")

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML (overrides other args)"
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        train_yolo(**config)
    else:
        train_yolo(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            name=args.name,
            device=args.device
        )


if __name__ == "__main__":
    main()
