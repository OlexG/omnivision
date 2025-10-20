#!/bin/bash
# Quick training script for drone detection

echo "Starting YOLOv8 training on drone-vs-bird dataset..."
echo "Using yolov8n (nano) model for fast training"
echo ""

python src/models/train.py \
    --data data/processed/drone-vs-bird-binary/data.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --batch 16 \
    --imgsz 640 \
    --name drone_detection_quick

echo ""
echo "Training complete! Model saved to models/checkpoints/drone_detection_quick/weights/best.pt"
