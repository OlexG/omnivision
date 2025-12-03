"""
Prepare binary detection dataset (drone vs not-drone) for evaluation.
This should match the dataset your detection model was trained on.
"""

import os
import shutil
from pathlib import Path
import yaml


def create_binary_detection_dataset(
    source_dataset: str,
    output_dataset: str,
    drone_class: str = "drone"
):
    """
    Create binary dataset: drone (0) vs not-drone (1).
    Relabels all non-drone classes as "not-drone".
    
    Args:
        source_dataset: Path to source YOLOv8 dataset
        output_dataset: Path to output binary dataset
        drone_class: Name of the drone class in source dataset
    """
    source_path = Path(source_dataset)
    output_path = Path(output_dataset)
    
    print(f"\n{'='*60}")
    print("CREATING BINARY DETECTION DATASET")
    print('='*60)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    # Read source data.yaml
    data_yaml = source_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"\nOriginal classes: {class_names}")
    
    # Find drone class index
    if isinstance(class_names, list):
        drone_idx = class_names.index(drone_class) if drone_class in class_names else None
    elif isinstance(class_names, dict):
        drone_idx = None
        for idx, name in class_names.items():
            if name == drone_class:
                drone_idx = idx
                break
    else:
        raise ValueError("Unexpected class names format")
    
    if drone_idx is None:
        raise ValueError(f"Drone class '{drone_class}' not found in dataset")
    
    print(f"Drone class index: {drone_idx}")
    
    # Process each split
    splits = ["train", "valid", "test"]
    stats = {}
    
    for split in splits:
        split_path = source_path / split
        if not split_path.exists():
            print(f"\n⚠ Split '{split}' not found, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        out_images = output_path / split / "images"
        out_labels = output_path / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        if not images_path.exists() or not labels_path.exists():
            print(f"  ⚠ Missing images or labels directory")
            continue
        
        processed_count = 0
        
        # Process each label file
        for label_file in labels_path.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Relabel: drone=0, everything else=1
            new_lines = []
            has_drone = False
            has_not_drone = False
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = int(parts[0])
                if class_id == drone_idx:
                    parts[0] = '0'  # drone
                    has_drone = True
                else:
                    parts[0] = '1'  # not-drone
                    has_not_drone = True
                
                new_lines.append(' '.join(parts) + '\n')
            
            # Copy image
            image_name = label_file.stem
            image_copied = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_file = images_path / f"{image_name}{ext}"
                if image_file.exists():
                    shutil.copy2(image_file, out_images / f"{image_name}{ext}")
                    image_copied = True
                    break
            
            if not image_copied:
                print(f"  ⚠ Image not found for {label_file.name}")
                continue
            
            # Write relabeled annotations
            with open(out_labels / label_file.name, 'w') as f:
                f.writelines(new_lines)
            
            processed_count += 1
        
        stats[split] = processed_count
        print(f"  ✓ Processed {processed_count} images")
    
    # Create binary data.yaml
    new_data_config = {
        'names': ['drone', 'not-drone'],
        'nc': 2,
        'train': '../train/images',
        'val': '../valid/images',
    }
    
    if 'test' in stats:
        new_data_config['test'] = '../test/images'
    
    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(new_data_config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print("DATASET CREATED")
    print('='*60)
    print(f"Output: {output_path}")
    print(f"Classes: ['drone', 'not-drone']")
    print(f"\nSplit statistics:")
    for split, count in stats.items():
        print(f"  {split}: {count} images")
    print('='*60)
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create binary detection dataset for evaluation"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source detection dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/detection_binary",
        help="Output path for binary dataset"
    )
    parser.add_argument(
        "--drone-class",
        type=str,
        default="drone",
        help="Name of drone class in source dataset"
    )
    
    args = parser.parse_args()
    
    output_path = create_binary_detection_dataset(
        source_dataset=args.source,
        output_dataset=args.output,
        drone_class=args.drone_class
    )
    
    print(f"\n{'='*60}")
    print("NEXT STEP: EVALUATION")
    print('='*60)
    print(f"\nNow run evaluation with:")
    print(f"\n  python src/evaluate.py \\")
    print(f"      --model models/checkpoints/drone_detection/weights/best_40epochs.pt \\")
    print(f"      --data {output_path}/data.yaml \\")
    print(f"      --classifier models/checkpoints/drone_classifier/best_40epochs.pt \\")
    print(f"      --eval-name my_evaluation")
    print()


if __name__ == "__main__":
    main()