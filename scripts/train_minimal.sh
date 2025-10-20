#!/bin/bash
# Minimal training for quick testing (uses tiny model and small subset)

echo "Starting minimal YOLOv8 training (very fast)..."
echo "Using yolov8n (nano) with reduced settings for speed"
echo ""

python src/models/train.py \
    --data data/processed/drone-vs-bird-binary/data.yaml \
    --model yolov8n.pt \
    --epochs 3 \
    --batch 8 \
    --imgsz 320 \
    --name drone_detection_minimal

echo ""
echo "Training complete! Model saved to models/checkpoints/drone_detection_minimal/weights/best.pt"
