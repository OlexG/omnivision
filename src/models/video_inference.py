"""
Video inference utility for the YOLOv8 drone detection model.
Streams frames from a video source, runs inference, and overlays detections.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
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
    primary_class: str = "drone"
) -> np.ndarray:
    annotated = frame.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls, str(cls))

        color = (0, 255, 0) if class_name == primary_class else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {conf:.2%}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = max(y1 - label_height - baseline, 0)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width, label_top + label_height + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_top + label_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
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
) -> None:
    """
    Run YOLOv8 inference on a video file or camera stream and draw bounding boxes.
    When displaying a video file, interactive controls allow seeking and pausing.
    """
    model = YOLO(model_path)

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
            annotated = _draw_detections(frame, result, model.names)
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
    )


if __name__ == "__main__":
    main()
