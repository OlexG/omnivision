#!/bin/bash
# Full training script with config file

echo "Starting YOLOv8 training with full configuration..."
echo ""

python src/models/train.py \
    --config models/configs/train_config.yaml

echo ""
echo "Training complete! Check models/checkpoints/ for results"
