"""
Extract drone crops from downloaded dataset using existing detection model.
This script processes images with ground truth labels to create a classification dataset.

Usage:
    # Using ground truth labels (faster, more accurate)
    python src/data/prepare_classifier_crops.py \
        --source data/part2/raw \
        --output data/part2/classification \
        --use-labels

    # Using detection model to auto-crop (if labels don't exist)
    python src/data/prepare_classifier_crops.py \
        --source data/part2/raw \
        --output data/part2/classification \
        --detection-model models/checkpoints/drone_detection_minimal/weights/best.pt
"""

import os
import cv2
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from ultralytics import YOLO


def read_yolo_label(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """
    Read YOLO format label file and convert to bounding boxes.

    Args:
        label_path: Path to .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of dicts with 'class_id' and 'bbox' [x1, y1, x2, y2]
    """
    detections = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert from YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                detections.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2]
                })

    return detections


def extract_crops_from_labels(
    source_dir: str,
    output_dir: str,
    class_names: Dict[int, str],
    min_crop_size: int = 32,
    padding: int = 10
) -> Dict[str, int]:
    """
    Extract crops using ground truth labels from YOLO dataset.

    Args:
        source_dir: Root directory of YOLO dataset
        output_dir: Output directory for classification dataset
        class_names: Dict mapping class IDs to class names
        min_crop_size: Minimum crop size (width or height) in pixels
        padding: Extra pixels to add around bounding box

    Returns:
        Dict with statistics (crops per class)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Find train, valid, test directories
    splits = ['train', 'valid', 'test']
    stats = {split: {name: 0 for name in class_names.values()} for split in splits}

    print("\n" + "="*60)
    print("EXTRACTING CROPS FROM LABELS")
    print("="*60)

    for split in splits:
        split_img_dir = source_path / split / 'images'
        split_label_dir = source_path / split / 'labels'

        if not split_img_dir.exists():
            print(f"\n⚠ Skipping {split} - directory not found")
            continue

        # Create output directories for this split
        for class_name in class_names.values():
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        # Process all images in this split
        image_files = list(split_img_dir.glob('*.jpg')) + \
                     list(split_img_dir.glob('*.jpeg')) + \
                     list(split_img_dir.glob('*.png'))

        print(f"\nProcessing {split} split ({len(image_files)} images)...")

        for img_path in tqdm(image_files, desc=f"  {split}"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            img_height, img_width = image.shape[:2]

            # Find corresponding label file
            label_path = split_label_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Read detections from label
            detections = read_yolo_label(str(label_path), img_width, img_height)

            # Extract crops
            for det_idx, detection in enumerate(detections):
                class_id = detection['class_id']
                x1, y1, x2, y2 = detection['bbox']

                # Skip invalid crops
                crop_width = x2 - x1
                crop_height = y2 - y1

                if crop_width < min_crop_size or crop_height < min_crop_size:
                    continue

                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_width, x2 + padding)
                y2 = min(img_height, y2 + padding)

                # Extract crop
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Save crop
                class_name = class_names.get(class_id, f"class_{class_id}")
                output_crop_dir = output_path / split / class_name
                crop_filename = f"{img_path.stem}_crop{det_idx}.jpg"
                crop_path = output_crop_dir / crop_filename

                cv2.imwrite(str(crop_path), crop)
                stats[split][class_name] += 1

    return stats


def extract_crops_from_detection(
    source_dir: str,
    output_dir: str,
    detection_model_path: str,
    class_names: List[str],
    conf_threshold: float = 0.5,
    min_crop_size: int = 32,
    padding: int = 10
) -> Dict[str, int]:
    """
    Extract crops using detection model predictions.
    Useful when ground truth labels don't exist or are in wrong format.

    Args:
        source_dir: Directory containing images
        output_dir: Output directory for classification dataset
        detection_model_path: Path to YOLOv8 detection model
        class_names: List of class names for manual labeling
        conf_threshold: Confidence threshold for detections
        min_crop_size: Minimum crop size (width or height) in pixels
        padding: Extra pixels to add around bounding box

    Returns:
        Dict with statistics
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Load detection model
    print(f"\nLoading detection model from {detection_model_path}...")
    model = YOLO(detection_model_path)

    # Create output directory for unlabeled crops
    unlabeled_dir = output_path / 'unlabeled'
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("EXTRACTING CROPS FROM DETECTION MODEL")
    print("="*60)
    print("\n⚠ Note: Crops will be saved as 'unlabeled' and need manual sorting")
    print(f"   into class folders: {', '.join(class_names)}\n")

    # Find all images
    image_files = list(source_path.rglob('*.jpg')) + \
                 list(source_path.rglob('*.jpeg')) + \
                 list(source_path.rglob('*.png'))

    total_crops = 0

    print(f"Processing {len(image_files)} images...")

    for img_path in tqdm(image_files, desc="  Detecting and cropping"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_height, img_width = image.shape[:2]

        # Run detection
        results = model.predict(img_path, conf=conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            continue

        # Extract crops from detections
        for det_idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])

            # Skip small crops
            crop_width = x2 - x1
            crop_height = y2 - y1

            if crop_width < min_crop_size or crop_height < min_crop_size:
                continue

            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_width, x2 + padding)
            y2 = min(img_height, y2 + padding)

            # Extract crop
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Save crop with confidence in filename for sorting
            crop_filename = f"{img_path.stem}_crop{det_idx}_conf{confidence:.2f}.jpg"
            crop_path = unlabeled_dir / crop_filename

            cv2.imwrite(str(crop_path), crop)
            total_crops += 1

    print(f"\n✓ Extracted {total_crops} crops to {unlabeled_dir}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review crops in {unlabeled_dir}")
    print(f"   2. Create class folders: {output_path / 'train' / '<class_name>'}")
    print(f"   3. Manually sort crops into appropriate class folders")
    print(f"   4. Split into train/val/test sets (~70%/15%/15%)")

    return {'unlabeled': total_crops}


def print_statistics(stats: Dict, output_dir: str):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    total_all = 0

    for split in ['train', 'valid', 'test']:
        if split in stats:
            print(f"\n{split.upper()} SET:")
            split_total = 0

            for class_name, count in sorted(stats[split].items()):
                if count > 0:
                    print(f"  {class_name:20s}: {count:5d} images")
                    split_total += count

            print(f"  {'Total':20s}: {split_total:5d} images")
            total_all += split_total

    if 'unlabeled' in stats:
        print(f"\nUNLABELED: {stats['unlabeled']} crops (need manual sorting)")
        total_all = stats['unlabeled']

    print(f"\n{'GRAND TOTAL':20s}: {total_all:5d} images")
    print("="*60)

    print(f"\n✓ Classification dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract drone crops for classification training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/part2/raw",
        help="Source directory (YOLO dataset or images)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/part2/classification",
        help="Output directory for classification dataset"
    )
    parser.add_argument(
        "--use-labels",
        action="store_true",
        help="Use ground truth labels from YOLO dataset (recommended)"
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        help="Path to detection model (if not using labels)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection model"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum crop size (width or height) in pixels"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding around bounding box in pixels"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.use_labels and not args.detection_model:
        parser.error("Either --use-labels or --detection-model must be specified")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    # Read class names from data.yaml if it exists
    data_yaml_path = source_path / "data.yaml"
    class_names = {}

    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            data_info = yaml.safe_load(f)

        if 'names' in data_info:
            class_names = data_info['names']
            print(f"\nFound {len(class_names)} classes in data.yaml:")
            for idx, name in class_names.items():
                print(f"  {idx}: {name}")

    # Extract crops
    if args.use_labels:
        if not class_names:
            raise ValueError(
                "No class names found in data.yaml. "
                "Cannot use labels without class mapping."
            )

        stats = extract_crops_from_labels(
            source_dir=args.source,
            output_dir=args.output,
            class_names=class_names,
            min_crop_size=args.min_size,
            padding=args.padding
        )
    else:
        # Use default class names if not found
        if not class_names:
            class_names = [
                'quadcopter',
                'fixed_wing',
                'fpv',
                'commercial',
                'military',
                'helicopter'
            ]
        else:
            class_names = list(class_names.values())

        stats = extract_crops_from_detection(
            source_dir=args.source,
            output_dir=args.output,
            detection_model_path=args.detection_model,
            class_names=class_names,
            conf_threshold=args.conf,
            min_crop_size=args.min_size,
            padding=args.padding
        )

    # Print statistics
    print_statistics(stats, args.output)


if __name__ == "__main__":
    main()
