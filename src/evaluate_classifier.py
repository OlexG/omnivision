import argparse
from pathlib import Path
import torch
import json
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def load_classifier(classifier_path, class_data_dir):
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=False)

    # Load class names from dataset structure
    train_dir = Path(class_data_dir) / "train"
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    num_classes = len(class_names)

    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, preprocess, class_names


def evaluate(model, preprocess, class_names, test_dir, batch_size=32):
    dataset = datasets.ImageFolder(
        root=test_dir,
        transform=preprocess
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs)
            preds = out.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    # Metrics
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    return report, cm


def save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True, help="Path to classifier .pt file")
    parser.add_argument("--class-data", required=True, help="Path to classification dataset (data/classification)")
    parser.add_argument("--eval-name", default="classifier_only_eval")
    args = parser.parse_args()

    eval_dir = Path(f"runs/evaluation/{args.eval_name}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.class_data) / "test"

    model, preprocess, class_names = load_classifier(
        args.classifier,
        args.class_data
    )

    print("Evaluating classifier on labeled test set...")

    report, cm = evaluate(model, preprocess, class_names, test_dir)

    # Save JSON
    with open(eval_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Save confusion matrix plot
    save_confusion_matrix(cm, class_names, eval_dir / "confusion_matrix.png")

    print("\n✓ Classifier-only evaluation complete.")
    print(f"Results saved to {eval_dir}\n")


if __name__ == "__main__":
    main()
