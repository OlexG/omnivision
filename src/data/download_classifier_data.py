"""
Download and prepare drone classification dataset from Roboflow.
This script downloads a multi-class drone dataset that can be either:
1. Classification dataset (already cropped images organized by class)
2. Object detection dataset (full images with labels - needs cropping)

Usage:
    # Download classification dataset (already cropped)
    python src/data/download_classifier_data.py \
        --workspace oleksandr-gorpynich \
        --project drone-detect-suvzw-gptrh \
        --version 1 \
        --format folder

    # Download object detection dataset (needs cropping)
    python src/data/download_classifier_data.py \
        --workspace paresh-makwana \
        --project drone-detect-suvzw \
        --version 1 \
        --format yolov8
"""

import os
import argparse
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv
import yaml
import shutil


def count_images_in_dir(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob('*.jpg'))) + \
           len(list(directory.glob('*.jpeg'))) + \
           len(list(directory.glob('*.png')))


def analyze_dataset_structure(dataset_path: Path) -> dict:
    """
    Analyze downloaded dataset to determine if it's classification or detection format.

    Returns:
        dict with 'type' (classification/detection) and 'structure' details
    """
    # Check for data.yaml (detection format)
    data_yaml = dataset_path / "data.yaml"

    # Check for train/valid/test directories
    train_dir = dataset_path / "train"
    valid_dir = dataset_path / "valid"
    test_dir = dataset_path / "test"

    structure = {
        'type': None,
        'train_dir': train_dir.exists(),
        'valid_dir': valid_dir.exists(),
        'test_dir': test_dir.exists(),
        'has_yaml': data_yaml.exists(),
        'classes': []
    }

    # Check if train directory has subdirectories (classification format)
    if train_dir.exists():
        subdirs = [d for d in train_dir.iterdir() if d.is_dir() and d.name not in ['images', 'labels']]

        if subdirs:
            # Classification format: train/<class_name>/images
            structure['type'] = 'classification'
            structure['classes'] = [d.name for d in subdirs]
            print(f"\n✓ Detected CLASSIFICATION dataset format")
            print(f"  Classes found: {', '.join(structure['classes'])}")

        elif (train_dir / 'images').exists() and (train_dir / 'labels').exists():
            # Object detection format: train/images/*.jpg, train/labels/*.txt
            structure['type'] = 'detection'
            if data_yaml.exists():
                with open(data_yaml, 'r') as f:
                    data_info = yaml.safe_load(f)
                    if 'names' in data_info:
                        structure['classes'] = list(data_info['names'].values())
            print(f"\n✓ Detected OBJECT DETECTION dataset format")
            print(f"  Needs cropping for classification training")

    return structure


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    format: str = "yolov8",
    output_dir: str = "data/part2/raw"
):
    """
    Download dataset from Roboflow Universe.

    Args:
        workspace: Roboflow workspace name (e.g., 'oleksandr-gorpynich')
        project: Project name (e.g., 'drone-detect-suvzw-gptrh')
        version: Dataset version number
        format: Download format ('yolov8', 'folder', 'coco', 'pascal-voc')
        output_dir: Directory to save the dataset

    Returns:
        dict with dataset_path and structure info
    """
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in .env file. "
            "Please add it: ROBOFLOW_API_KEY=your_api_key_here\n"
            "Get your API key from: https://app.roboflow.com/settings/api"
        )

    # Initialize Roboflow
    print(f"\n" + "="*60)
    print("DOWNLOADING FROM ROBOFLOW")
    print("="*60)
    print(f"\nWorkspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    print(f"Format: {format}")

    rf = Roboflow(api_key=api_key)

    # Get project
    print(f"\nAccessing project...")
    project_obj = rf.workspace(workspace).project(project)

    # Get specific version
    print(f"Getting version {version}...")
    dataset = project_obj.version(version)

    # Download dataset
    print(f"\nDownloading dataset...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Roboflow appends version to path, so we download to parent and then move
    temp_location = str(output_path.parent / "temp_download")

    dataset.download(
        model_format=format,
        location=temp_location
    )

    # Find the downloaded directory (Roboflow creates projectname-version folder)
    temp_path = Path(temp_location)
    downloaded_dirs = [d for d in temp_path.iterdir() if d.is_dir()]

    if downloaded_dirs:
        actual_download = downloaded_dirs[0]
        # Move contents to our desired location
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(actual_download), str(output_path))
        # Clean up temp directory
        if temp_path.exists():
            shutil.rmtree(temp_path)

    print(f"\n✓ Dataset downloaded successfully to {output_dir}")

    # Analyze dataset structure
    structure = analyze_dataset_structure(output_path)

    # Print detailed information
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)

    print(f"\nDataset type: {structure['type'].upper() if structure['type'] else 'UNKNOWN'}")
    print(f"Classes ({len(structure['classes'])}):")
    for name in structure['classes']:
        print(f"  - {name}")

    # Count images per split
    print(f"\nDataset splits:")
    for split in ['train', 'valid', 'test']:
        split_dir = output_path / split
        if structure['type'] == 'classification':
            # Count images in all class subdirectories
            total = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = count_images_in_dir(class_dir)
                    total += count
            print(f"  {split:6s}: {total:5d} images")
        elif structure['type'] == 'detection':
            img_dir = split_dir / 'images'
            count = count_images_in_dir(img_dir)
            print(f"  {split:6s}: {count:5d} images")

    print("="*60 + "\n")

    return {
        'path': str(output_path),
        'structure': structure
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download drone classification dataset from Roboflow"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="oleksandr-gorpynich",
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="drone-detect-suvzw-gptrh",
        help="Roboflow project name"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version number"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="folder",
        choices=["yolov8", "yolov5", "coco", "pascal-voc", "folder"],
        help="Download format (use 'folder' for classification datasets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/part2",
        help="Output directory for downloaded dataset"
    )

    args = parser.parse_args()

    # Download dataset
    result = download_roboflow_dataset(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        format=args.format,
        output_dir=args.output_dir
    )

    dataset_path = result['path']
    structure = result['structure']

    print(f"\n✓ Dataset ready at: {dataset_path}")

    # Provide next steps based on dataset type
    if structure['type'] == 'classification':
        print(f"\n{'='*60}")
        print("✓ READY FOR TRAINING!")
        print("="*60)
        print("\nThis is a classification dataset (already cropped).")
        print("You can proceed directly to training the classifier:")
        print(f"\n  python src/models/train_classifier.py \\")
        print(f"      --data {dataset_path} \\")
        print(f"      --config models/configs/classifier_config.yaml")

    elif structure['type'] == 'detection':
        print(f"\n{'='*60}")
        print("NEXT STEP: EXTRACT CROPS")
        print("="*60)
        print("\nThis is a detection dataset (full images with labels).")
        print("Run this command to extract drone crops:")
        print(f"\n  python src/data/prepare_classifier_crops.py \\")
        print(f"      --source {dataset_path} \\")
        print(f"      --output {Path(dataset_path).parent / 'classification'} \\")
        print(f"      --use-labels")

    print("\n")


if __name__ == "__main__":
    main()
