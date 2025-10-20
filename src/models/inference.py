"""
YOLOv8 inference script for drone detection.
Run predictions on images and visualize results.
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np


def run_inference(
    model_path: str,
    source: str,
    save_dir: str = "results/predictions",
    conf: float = 0.25,
    iou: float = 0.45,
    show: bool = False,
    save: bool = True
):
    """
    Run YOLOv8 inference on images.

    Args:
        model_path: Path to trained model (.pt file)
        source: Path to image, directory, or video
        save_dir: Directory to save results
        conf: Confidence threshold
        iou: IOU threshold for NMS
        show: Display results
        save: Save annotated images
    """
    print(f"\n{'='*60}")
    print(f"Running YOLOv8 Inference")
    print(f"{'='*60}\n")

    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf}")
    print(f"IOU threshold: {iou}\n")

    # Load model
    model = YOLO(model_path)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save,
        project=str(save_path.parent),
        name=save_path.name,
        show=show
    )

    # Process and display results
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}\n")

    for i, result in enumerate(results):
        print(f"Image {i+1}: {result.path}")

        if len(result.boxes) == 0:
            print("  No detections")
        else:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                class_name = model.names[cls]

                print(f"  - {class_name}: {conf:.2%} confidence")
                print(f"    Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        print()

    if save:
        print(f"✓ Results saved to: {save_path}")
    print()

    return results


def annotate_image(
    image_path: str,
    model_path: str,
    output_path: str = None,
    conf: float = 0.25
):
    """
    Annotate a single image with detailed detection info.

    Args:
        image_path: Path to input image
        model_path: Path to trained model
        output_path: Path to save annotated image
        conf: Confidence threshold
    """
    # Load model and image
    model = YOLO(model_path)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Run inference
    results = model(img, conf=conf)[0]

    # Draw annotations
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Get class and confidence
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[cls]

        # Choose color based on class (green for drone, red for not-drone)
        color = (0, 255, 0) if class_name == "drone" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{class_name}: {confidence:.2%}"
        coords_text = f"[{x1},{y1},{x2},{y2}]"

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)

        # Draw text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, coords_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save or display
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"✓ Annotated image saved to: {output_path}")
    else:
        cv2.imshow("Detection Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for drone detection")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, directory, or video"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/predictions",
        help="Directory to save results"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results"
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        source=args.source,
        save_dir=args.save_dir,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=not args.no_save
    )


if __name__ == "__main__":
    main()
