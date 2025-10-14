# Omnivision

Vision-based drone detection system for identifying and classifying drones within 100-meter range.

## Setup

0. Install uv
Install uv by following [this link](https://docs.astral.sh/uv/getting-started/installation/)
i.e. running
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync
```

2. Configure API key in `.env`:
```bash
ROBOFLOW_API_KEY=your_api_key_here
```

3. Download and process datasets:
```bash
python src/data/collection.py    # Downloads raw datasets
python src/data/filter_class.py  # Filters to drone-only class
```

## Datasets

**Dataset 1:** [Drone vs Bird Object Detection](https://universe.roboflow.com/oleksandr-gorpynich/drone-vs-bird-object-detection-2fnnk/1)
- Source: Roboflow Universe
- Classes: Drone (filtered from bird/drone)
- License: CC BY 4.0
- Augmentation: 1% noise (applied in Roboflow)
- Train: 4,200 images
- Validation: 400 images
- Test: 199 images

**Dataset 2:** [Airborne Object Detection (AOD4)](https://universe.roboflow.com/oleksandr-gorpynich/airborne-object-detection-4-aod4-zaeoh/1)
- Source: Roboflow Universe
- Classes: Drone (filtered from airplane/bird/drone/helicopter)
- License: CC BY 4.0
- Train: 15,593 images
- Validation: 1,485 images
- Test: 742 images
