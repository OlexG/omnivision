"""
Unified YOLO + classifier evaluation.
- Evaluates YOLO detection (mAP, PR curves, confusion matrix)
- Runs classifier on YOLO-detected drones from validation set
- Compares predictions against ground truth labels
- Saves metrics, confusion matrix, subtype distribution, sample crops
"""

import argparse
import json
from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def evaluate_detection(model_path, data_yaml, eval_dir):
    """Evaluate YOLO detection performance."""
    print(f"\n{'='*60}")
    print("EVALUATING YOLO DETECTION")
    print('='*60)
    print(f"Model: {model_path}")

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, save_json=True, plots=True)

    det_metrics = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "class_map": metrics.box.maps.tolist() if hasattr(metrics.box, 'maps') else [],
        "class_names": model.names,
    }

    with open(eval_dir / "detection_metrics.json", "w") as f:
        json.dump(det_metrics, f, indent=4)

    print(f"\nDetection Results:")
    print(f"  mAP@50: {det_metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {det_metrics['mAP50-95']:.4f}")
    print(f"  Precision: {det_metrics['precision']:.4f}")
    print(f"  Recall: {det_metrics['recall']:.4f}")
    print("✓ YOLO evaluation complete")
    
    return model, det_metrics


def load_classifier(classifier_path, device='cpu', classification_data_path=None):
    print(f"\nLoading classifier from: {classifier_path}")

    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)

    # ------------------------------------------------------
    # 1. Determine class names
    # ------------------------------------------------------
    class_names = None

    # Case A — checkpoint contains class names
    for key in ["class_names", "classes"]:
        if key in checkpoint:
            class_names = checkpoint[key]
            break

    # Case B — infer from classification dataset
    if class_names is None and classification_data_path:
        train_dir = Path(classification_data_path) / "train"
        if train_dir.exists():
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            print(f"  ✓ Inferred classes from dataset: {class_names}")

    # Case C — infer from state_dict shape
    if class_names is None:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        num_classes = state_dict["classifier.1.weight"].shape[0]
        class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"  ⚠ Inferred classes from architecture: {class_names}")

    num_classes = len(class_names)

    # ------------------------------------------------------
    # 2. Build EfficientNet model
    # ------------------------------------------------------
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # ------------------------------------------------------
    # 3. Load weights (either wrapped or raw)
    # ------------------------------------------------------
    if "model_state_dict" in checkpoint:
        print("  ✓ Loading wrapped checkpoint (model_state_dict)")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("  ✓ Loading raw state_dict")
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    print(f"Classifier loaded with {num_classes} classes.")
    return model, preprocess, class_names


def get_ground_truth_from_labels(label_path, img_width, img_height, class_names_yolo):
    """
    Extract ground truth class from YOLO label file.
    Assumes single-class per image for simplicity.
    Returns the most common class in the image.
    """
    if not label_path.exists():
        return None
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    # Get all classes in the image
    classes = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_id = int(parts[0])
            classes.append(class_id)
    
    if not classes:
        return None
    
    # Return most common class
    # For drone classification, we typically have one object per image
    most_common_class_id = max(set(classes), key=classes.count)
    
    # Map class ID to class name
    if most_common_class_id < len(class_names_yolo):
        return class_names_yolo[most_common_class_id]
    
    return None


def evaluate_classifier_on_detections(
    yolo_model,
    classifier,
    preprocess,
    class_names,
    data_yaml,
    eval_dir,
    detection_conf=0.25,
    device='cpu',
):
    """
    Run classifier on YOLO detections and compare against ground truth.
    """
    print(f"\n{'='*60}")
    print("EVALUATING CLASSIFIER ON DETECTIONS")
    print('='*60)

    # Load validation images from YAML
    yaml_path = Path(data_yaml)
    with open(yaml_path) as f:
        import yaml
        data_cfg = yaml.safe_load(f)

    # Get validation image and label paths
    val_img_dir = (yaml_path.parent / data_cfg["val"]).resolve()
    val_label_dir = val_img_dir.parent / "labels"
    
    print(f"Validation images: {val_img_dir}")
    print(f"Validation labels: {val_label_dir}")

    # Storage for predictions and ground truth
    y_true = []
    y_pred = []
    y_true_names = []
    y_pred_names = []
    confidences = []
    
    output_stats = {
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "classified": 0,
        "matched_to_ground_truth": 0,
    }

    # Create directories for sample crops
    samples_dir = eval_dir / "classified_samples"
    samples_dir.mkdir(exist_ok=True)
    
    correct_dir = samples_dir / "correct"
    incorrect_dir = samples_dir / "incorrect"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)

    # Process each validation image
    img_paths = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    
    for img_path in img_paths:
        output_stats["total_images"] += 1
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_height, img_width = img.shape[:2]

        # Get ground truth from label file
        label_path = val_label_dir / f"{img_path.stem}.txt"
        gt_class = get_ground_truth_from_labels(
            label_path, 
            img_width, 
            img_height, 
            yolo_model.names
        )

        # Run YOLO detection
        results = yolo_model.predict(img, conf=detection_conf, verbose=False)[0]

        if len(results.boxes) > 0:
            output_stats["images_with_detections"] += 1

        # Process each detection
        for box in results.boxes:
            cls = int(box.cls[0])
            detected_class = yolo_model.names[cls]
            
            # Only process drone detections
            if detected_class != "drone":
                continue

            output_stats["total_detections"] += 1

            # Extract crop
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = img[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Classify the crop
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = preprocess(crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                out = classifier(crop_tensor)
                probs = torch.softmax(out, dim=1)[0]
                score, idx = probs.max(0)

            pred_class = class_names[idx.item()]
            confidence = score.item()
            
            output_stats["classified"] += 1

            # Compare to ground truth if available
            if gt_class is not None and gt_class in class_names:
                output_stats["matched_to_ground_truth"] += 1
                
                gt_idx = class_names.index(gt_class)
                pred_idx = idx.item()
                
                y_true.append(gt_idx)
                y_pred.append(pred_idx)
                y_true_names.append(gt_class)
                y_pred_names.append(pred_class)
                confidences.append(confidence)
                
                # Save example crop
                is_correct = (gt_class == pred_class)
                save_dir = correct_dir if is_correct else incorrect_dir
                save_path = save_dir / f"{img_path.stem}_gt{gt_class}_pred{pred_class}_conf{confidence:.2f}.jpg"
                cv2.imwrite(str(save_path), crop)

    # Calculate metrics if we have ground truth
    if len(y_true) > 0:
        print(f"\n{'='*60}")
        print("CLASSIFICATION METRICS")
        print('='*60)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names))
        )
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<20} {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}  {support[i]:>8.0f}")
        
        # Average metrics
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        print(f"\n{'Macro Average':<20} {avg_precision:>10.4f}  {avg_recall:>10.4f}  {avg_f1:>10.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        
        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "macro_precision": float(avg_precision),
            "macro_recall": float(avg_recall),
            "macro_f1": float(avg_f1),
            "per_class_metrics": {
                class_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }
                for i in range(len(class_names))
            },
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }
        
        with open(eval_dir / "classification_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save detailed classification report
        with open(eval_dir / "classification_report.txt", "w") as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, eval_dir)
        
        # Plot confidence distribution
        plot_confidence_distribution(confidences, y_true, y_pred, class_names, eval_dir)
        
    else:
        print("\n⚠ No ground truth labels found for classification evaluation")
        metrics = None

    # Save overall stats
    output_stats["classification_metrics"] = metrics
    with open(eval_dir / "evaluation_stats.json", "w") as f:
        json.dump(output_stats, f, indent=4)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total images processed: {output_stats['total_images']}")
    print(f"Images with detections: {output_stats['images_with_detections']}")
    print(f"Total drone detections: {output_stats['total_detections']}")
    print(f"Classified crops: {output_stats['classified']}")
    print(f"Matched to ground truth: {output_stats['matched_to_ground_truth']}")
    
    return output_stats, metrics

def evaluate_classifier_standalone(classifier, preprocess, class_names, classification_data_path, eval_dir, device='cpu'):
    """Evaluate classifier on its own validation set (folder-based structure)."""
    print(f"\n{'='*60}")
    print("EVALUATING CLASSIFIER (STANDALONE)")
    print('='*60)
    
    val_dir = Path(classification_data_path) / "valid"
    print(f"Validation directory: {val_dir}")
    
    y_true = []
    y_pred = []
    confidences = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = val_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠ Missing class folder: {class_name}")
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        print(f"  {class_name}: {len(images)} images")
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = classifier(img_tensor)
                probs = torch.softmax(out, dim=1)[0]
                score, idx = probs.max(0)
            
            y_true.append(class_idx)
            y_pred.append(idx.item())
            confidences.append(score.item())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    print(f"\nClassifier Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro Precision: {avg_precision:.4f}")
    print(f"  Macro Recall: {avg_recall:.4f}")
    print(f"  Macro F1: {avg_f1:.4f}")
    
    # Save metrics
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(avg_precision),
        "macro_recall": float(avg_recall),
        "macro_f1": float(avg_f1),
        "per_class_metrics": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    
    with open(eval_dir / "classifier_standalone_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, eval_dir)
    plot_confidence_distribution(confidences, y_true, y_pred, class_names, eval_dir)
    
    print("✓ Classifier evaluation complete")
    return metrics


def plot_confusion_matrix(cm, class_names, eval_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix (Raw Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix_counts.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences, y_true, y_pred, class_names, eval_dir):
    """Plot confidence score distributions for correct vs incorrect predictions."""
    correct_confidences = [c for c, t, p in zip(confidences, y_true, y_pred) if t == p]
    incorrect_confidences = [c for c, t, p in zip(confidences, y_true, y_pred) if t != p]
    
    plt.figure(figsize=(10, 6))
    
    if correct_confidences:
        plt.hist(correct_confidences, bins=20, alpha=0.6, label='Correct', color='green')
    if incorrect_confidences:
        plt.hist(incorrect_confidences, bins=20, alpha=0.6, label='Incorrect', color='red')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(eval_dir / "confidence_distribution.png", dpi=300)
    plt.close()


def plot_subtype_distribution(y_true, y_pred, class_names, eval_dir):
    """Plot distribution of predictions vs ground truth."""
    true_counts = [y_true.count(i) for i in range(len(class_names))]
    pred_counts = [y_pred.count(i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, true_counts, width, label='Ground Truth', alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8)
    
    ax.set_xlabel('Drone Type')
    ax.set_ylabel('Count')
    ax.set_title('Ground Truth vs Predicted Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(eval_dir / "subtype_distribution.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detection + classifier with ground truth comparison"
    )
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--classifier", required=True, help="Path to classifier checkpoint")
    parser.add_argument("--classification-data", default="data/part2", help="Path to classification dataset (for class names)")
    parser.add_argument("--eval-name", default="eval", help="Name for evaluation run")
    parser.add_argument("--detection-conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    
    args = parser.parse_args()

    # Setup
    eval_dir = Path(f"runs/evaluation/{args.eval_name}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("OMNIVISION EVALUATION")
    print('='*60)
    print(f"YOLO Model: {args.model}")
    print(f"Classifier: {args.classifier}")
    print(f"Data: {args.data}")
    print(f"Classification Data: {args.classification_data}")
    print(f"Output: {eval_dir}")
    print('='*60)

    # 1) Evaluate YOLO detection
    yolo_model, det_metrics = evaluate_detection(args.model, args.data, eval_dir)

    # 2) Load classifier
    device = args.device if torch.cuda.is_available() else 'cpu'
    classifier, preprocess, class_names = load_classifier(
        args.classifier, 
        device=device, 
        classification_data_path=args.classification_data
    )

    # 3) Evaluate classifier on detections with ground truth comparison
    stats, class_metrics = evaluate_classifier_on_detections(
        yolo_model,
        classifier,
        preprocess,
        class_names,
        args.data,
        eval_dir,
        detection_conf=args.detection_conf,
        device=device,
    )

    # 3) Evaluate classifier standalone
    class_metrics = evaluate_classifier_standalone(
        classifier,
        preprocess,
        class_names,
        args.classification_data,
        eval_dir,
        device=device,
    )

    # 4) Create visual summaries
    if class_metrics is not None:
        # Get predictions for plotting
        with open(eval_dir / "classifier_standalone_metrics.json", 'r') as f:
            metrics_data = json.load(f)
        
        # Reconstruct y_true and y_pred from confusion matrix
        cm = np.array(metrics_data['confusion_matrix'])
        y_true_reconstructed = []
        y_pred_reconstructed = []
        
        for true_idx in range(len(class_names)):
            for pred_idx in range(len(class_names)):
                count = cm[true_idx, pred_idx]
                y_true_reconstructed.extend([true_idx] * int(count))
                y_pred_reconstructed.extend([pred_idx] * int(count))
        
        plot_subtype_distribution(y_true_reconstructed, y_pred_reconstructed, class_names, eval_dir)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print('='*60)
    print(f"Results saved to: {eval_dir}")
    print(f"\nKey files:")
    print(f"  - detection_metrics.json")
    print(f"  - classification_metrics.json")
    print(f"  - classification_report.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - confidence_distribution.png")
    print(f"  - subtype_distribution.png")
    print(f"  - classified_samples/correct/")
    print(f"  - classified_samples/incorrect/")
    print('='*60 + "\n")


if __name__ == "__main__":
    main()