"""
Filter dataset to only include specific classes.
"""

import os
import shutil
from pathlib import Path
import yaml


def filter_yolo_dataset(
    dataset_path: str,
    output_path: str,
    keep_classes: list,
    splits: list = ["train", "valid", "test"]
):
    """
    Filter a YOLO dataset to only keep specified classes.

    Args:
        dataset_path: Path to original dataset
        output_path: Path to save filtered dataset
        keep_classes: List of class names to keep (e.g., ["drone"])
        splits: Dataset splits to process (default: ["train", "valid", "test"])
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Read data.yaml to get class mapping
    with open(dataset_path / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"Original classes: {class_names}")

    # Get indices of classes to keep
    keep_indices = [i for i, name in enumerate(class_names) if name in keep_classes]
    print(f"Keeping classes: {[class_names[i] for i in keep_indices]}")

    # Create new class mapping (reindexed starting from 0)
    new_class_names = [class_names[i] for i in keep_indices]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

    # Process each split
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Split '{split}' not found, skipping...")
            continue

        # Create output directories
        out_images = output_path / split / "images"
        out_labels = output_path / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        images_path = split_path / "images"
        labels_path = split_path / "labels"

        kept_count = 0
        removed_count = 0

        # Process each label file
        for label_file in labels_path.glob("*.txt"):
            # Read annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Filter annotations for desired classes
            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id in keep_indices:
                    # Remap class ID
                    parts[0] = str(index_mapping[class_id])
                    filtered_lines.append(' '.join(parts) + '\n')

            # Only keep images that have at least one annotation of the desired class
            if filtered_lines:
                # Copy image
                image_name = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    image_file = images_path / f"{image_name}{ext}"
                    if image_file.exists():
                        shutil.copy2(image_file, out_images / f"{image_name}{ext}")
                        break

                # Write filtered labels
                with open(out_labels / label_file.name, 'w') as f:
                    f.writelines(filtered_lines)

                kept_count += 1
            else:
                removed_count += 1

        print(f"Split '{split}': kept {kept_count} images, removed {removed_count} images")

    # Create new data.yaml
    new_data_config = {
        'names': new_class_names,
        'nc': len(new_class_names),
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images'
    }

    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(new_data_config, f, default_flow_style=False)

    print(f"\n✓ Filtered dataset saved to: {output_path}")
    print(f"New classes: {new_class_names}")


def create_binary_dataset(
    dataset_path: str,
    output_path: str,
    drone_class: str = "drone",
    splits: list = ["train", "valid", "test"]
):
    """
    Create binary dataset: drone (0) vs not-drone (1).
    Relabels all non-drone classes as "not-drone".
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Read data.yaml
    with open(dataset_path / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"Original classes: {class_names}")

    drone_idx = class_names.index(drone_class)
    print(f"Drone class index: {drone_idx}")

    # Process each split
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Split '{split}' not found, skipping...")
            continue

        out_images = output_path / split / "images"
        out_labels = output_path / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        images_path = split_path / "images"
        labels_path = split_path / "labels"

        for label_file in labels_path.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Relabel: drone=0, everything else=1
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id == drone_idx:
                    parts[0] = '0'  # drone
                else:
                    parts[0] = '1'  # not-drone
                new_lines.append(' '.join(parts) + '\n')

            # Copy image
            image_name = label_file.stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_file = images_path / f"{image_name}{ext}"
                if image_file.exists():
                    shutil.copy2(image_file, out_images / f"{image_name}{ext}")
                    break

            # Write relabeled annotations
            with open(out_labels / label_file.name, 'w') as f:
                f.writelines(new_lines)

        print(f"Split '{split}': processed")

    # Create data.yaml with binary classes
    new_data_config = {
        'names': ['drone', 'not-drone'],
        'nc': 2,
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images'
    }

    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(new_data_config, f, default_flow_style=False)

    print(f"\n✓ Binary dataset saved to: {output_path}")


if __name__ == "__main__":
    from pathlib import Path

    # Ensure data/processed exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Process first dataset: Drone vs Bird (binary)
    print("\n=== Processing Dataset 1: Drone vs Bird (Binary) ===")
    create_binary_dataset(
        dataset_path="data/raw/drone-vs-bird-object-detection-1",
        output_path="data/processed/drone-vs-bird-binary",
        drone_class="drone",
        splits=["train", "valid", "test"]
    )

    # Process second dataset: Airborne (binary)
    print("\n=== Processing Dataset 2: Airborne (Binary) ===")
    create_binary_dataset(
        dataset_path="data/raw/Airborne-Object-Detection-4-AOD4-1",
        output_path="data/processed/airborne-binary",
        drone_class="drone",
        splits=["train", "valid", "test"]
    )

    print("\n✓ All datasets processed to data/processed/")
