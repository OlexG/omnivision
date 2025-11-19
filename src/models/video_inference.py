"""
Video inference utility for the YOLOv8 drone detection model.
Streams frames from a video source, runs inference, and overlays detections.
Supports two-stage pipeline: detection + classification.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO


def _format_source(source: str) -> int | str:
    """Allow numeric camera indices while keeping support for file paths."""
    try:
        return int(source)
    except ValueError:
        return source


def _draw_detections(
    frame: np.ndarray,
    result,
    class_names: dict[int, str],
    primary_class: str = "drone",
    classifications: Optional[dict[int, tuple[str, float]]] = None,
) -> np.ndarray:
    """
    Draw detection bounding boxes with optional classification labels.
    
    Args:
        frame: Input frame
        result: YOLO detection result
        class_names: Mapping of class IDs to names
        primary_class: Primary class name (e.g., "drone")
        classifications: Optional dict mapping box index to (class_name, confidence)
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls, str(cls))

        color = (0, 255, 0) if class_name == primary_class else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label with detection and classification info
        if classifications and idx in classifications:
            drone_type, cls_conf = classifications[idx]
            label = f"{drone_type}\nDet: {conf:.1%} | Cls: {cls_conf:.1%}"
        else:
            label = f"{class_name}: {conf:.2%}"

        # Calculate text size for multi-line label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 20
        
        if "\n" in label:
            lines = label.split("\n")
            max_width = 0
            for line in lines:
                (label_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, label_width)
            label_height = len(lines) * line_height
            label_width = max_width
        else:
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            label_height += baseline

        label_top = max(y1 - label_height - 5, 0)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width + 10, label_top + label_height + 5),
            color,
            -1,
        )
        
        # Draw text (handle multi-line)
        if "\n" in label:
            y_offset = label_top + line_height
            for line in lines:
                cv2.putText(
                    annotated,
                    line,
                    (x1 + 5, y_offset),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
                y_offset += line_height
        else:
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_top + label_height),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return annotated


def run_video_inference(
    model_path: str,
    source: str,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    imgsz: int = 640,
    display: bool = True,
    output_path: Optional[str] = None,
    window_name: str = "YOLOv8 Drone Detection",
    classifier_path: Optional[str] = None,
    classification_conf: float = 0.3,
) -> None:
    """
    Run YOLOv8 inference on a video file or camera stream and draw bounding boxes.
    When displaying a video file, interactive controls allow seeking and pausing.
    
    Args:
        model_path: Path to detection model
        source: Video source (file path or camera index)
        conf: Detection confidence threshold
        iou: IOU threshold for NMS
        device: Device for inference ('cpu', '0', etc.)
        imgsz: Image size for inference
        display: Whether to display video
        output_path: Optional path to save output video
        window_name: Window title for display
        classifier_path: Optional path to classification model
        classification_conf: Minimum detection confidence to run classification
    """
    model = YOLO(model_path)
    
    # Load classification model if provided
    classifier = None
    classifier_device = None
    preprocess = None
    classifier_class_names = None
    
    if classifier_path:
        print(f"Loading classification model from {classifier_path}...")
        checkpoint = torch.load(classifier_path, map_location='cpu')
        
        # Get number of classes from checkpoint
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        elif 'class_names' in checkpoint:
            num_classes = len(checkpoint['class_names'])
            classifier_class_names = checkpoint['class_names']
        else:
            # Try to infer from model state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Look for classifier head
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
            else:
                num_classes = 10  # Default to 10 classes
        
        # Create model
        classifier = efficientnet_b0(weights=None)
        classifier.classifier[1] = torch.nn.Linear(
            classifier.classifier[1].in_features, num_classes
        )
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        
        # Setup device
        if device and device != 'cpu':
            try:
                device_idx = int(device)
                classifier_device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
            except ValueError:
                classifier_device = torch.device('cpu')
        else:
            classifier_device = torch.device('cpu')
        
        classifier = classifier.to(classifier_device)
        
        # Setup preprocessing (ImageNet normalization)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Get class names if available
        if classifier_class_names is None:
            if 'class_names' in checkpoint:
                classifier_class_names = checkpoint['class_names']
            else:
                # Default class names based on the notebook
                classifier_class_names = [
                    'Cinewhoop', 'DJI FPV', 'DJI Mavic', 'DJI Phantom', 'Fixed wing',
                    'Hexacopter', 'Octocopter', 'Pluto Mini Drone', 'Quadcopter', 'VTOL'
                ][:num_classes]
        
        print(f"✓ Classification model loaded ({num_classes} classes)")
        print(f"  Classes: {', '.join(classifier_class_names)}")
        print(f"  Device: {classifier_device}")

    capture = cv2.VideoCapture(_format_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    is_seekable = (
        display
        and isinstance(source, str)
        and Path(source).is_file()
        and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    )

    writer = None
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_time = time.time()
    frames_processed = 0
    paused = False
    advance_step = False
    last_annotated: Optional[np.ndarray] = None

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_seekable else 0
    trackbar_seek_requested: Optional[int] = None
    updating_trackbar = False
    if is_seekable and total_frames > 0:

        def on_trackbar(val: int) -> None:
            nonlocal trackbar_seek_requested, updating_trackbar, paused, frames_processed, start_time
            if updating_trackbar:
                return
            trackbar_seek_requested = val
            paused = True  # freeze playback while seeking
            frames_processed = 0
            start_time = time.time()

        cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 0), on_trackbar)

    try:
        while True:
            if paused and not advance_step and trackbar_seek_requested is None:
                if display and last_annotated is not None:
                    cv2.imshow(window_name, last_annotated)
                if display:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = False
                    elif key == ord("n"):
                        advance_step = True
                    elif is_seekable and key in (81, ord("a")):  # left arrow or 'a'
                        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        seek_target = max(current_frame - 30, 0)
                        trackbar_seek_requested = seek_target
                    elif is_seekable and key in (83, ord("d")):  # right arrow or 'd'
                        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        seek_target = min(current_frame + 30, total_frames - 1)
                        trackbar_seek_requested = seek_target
                    elif key == ord("r"):
                        frames_processed = 0
                        start_time = time.time()
                else:
                    time.sleep(0.03)
                continue

            if trackbar_seek_requested is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, trackbar_seek_requested)
                advance_step = True
                trackbar_seek_requested = None

            ret, frame = capture.read()
            if not ret:
                break

            inference_start = time.time()
            result = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]
            
            # Run classification on detected drones
            classifications = None
            if classifier and len(result.boxes) > 0:
                classifications = {}
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                for idx, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    det_conf = float(box.conf[0])
                    class_name = model.names.get(cls_id, str(cls_id))
                    
                    # Only classify if it's a "drone" detection above threshold
                    if class_name == "drone" and det_conf >= classification_conf:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Ensure coordinates are within bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Extract crop
                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                            try:
                                # Preprocess and classify
                                crop_tensor = preprocess(crop).unsqueeze(0).to(classifier_device)
                                with torch.no_grad():
                                    output = classifier(crop_tensor)
                                    probs = torch.softmax(output, dim=1)[0]
                                    cls_conf, pred_idx = probs.max(0)
                                
                                drone_type = classifier_class_names[pred_idx.item()]
                                classifications[idx] = (drone_type, cls_conf.item())
                            except Exception as e:
                                # Skip classification if there's an error
                                pass
            
            annotated = _draw_detections(frame, result, model.names, classifications=classifications)
            inference_dt = time.time() - inference_start

            frames_processed += 1
            elapsed = max(time.time() - start_time, 1e-6)
            avg_fps = frames_processed / elapsed

            cv2.putText(
                annotated,
                f"Inference FPS: {1.0 / max(inference_dt, 1e-6):.1f} | Avg FPS: {avg_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if display:
                last_annotated = annotated
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("n"):
                    advance_step = True
                elif key == ord("r"):
                    frames_processed = 0
                    start_time = time.time()
                elif is_seekable and key in (81, ord("a")):
                    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                    trackbar_seek_requested = max(current_frame - 30, 0)
                elif is_seekable and key in (83, ord("d")):
                    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                    trackbar_seek_requested = min(current_frame + 30, total_frames - 1)

                if is_seekable:
                    current_frame = max(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
                    updating_trackbar = True
                    cv2.setTrackbarPos("Frame", window_name, current_frame)
                    updating_trackbar = False

            if writer:
                writer.write(annotated)

            advance_step = False
    finally:
        capture.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyWindow(window_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 drone detection on a video file or webcam feed."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or camera index (e.g., '0').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for non-max suppression.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g., 'cpu', '0').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resize frames to this dimension before inference.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable on-screen display of the annotated video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save the annotated video.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="YOLOv8 Drone Detection",
        help="Window title for live display.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Optional path to classification model for drone type classification.",
    )
    parser.add_argument(
        "--classification-conf",
        type=float,
        default=0.3,
        help="Minimum detection confidence to run classification (default: 0.3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_video_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        display=not args.no_display,
        output_path=args.output,
        window_name=args.window_name,
        classifier_path=args.classifier,
        classification_conf=args.classification_conf,
    )


if __name__ == "__main__":
    main()
