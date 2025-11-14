# Drone Classification Dataset (Part 2)

This directory contains the dataset for training the Stage 2 drone type classifier.

## Quick Start

### Step 1: Set up Roboflow API Key

Get your API key from: https://app.roboflow.com/settings/api

Add it to your `.env` file in the project root:

```bash
echo "ROBOFLOW_API_KEY=your_api_key_here" >> .env
```

### Step 2: Download the Dataset

Run the download script:

```bash
python src/data/download_classifier_data.py
```

This will:
- Download your dataset from Roboflow (oleksandr-gorpynich/drone-detect-suvzw-gptrh)
- Automatically detect if it's a classification dataset (already cropped) or detection dataset (needs cropping)
- Save it to `data/part2/`
- Display statistics about classes and image counts

### Step 3: Follow the Instructions

After downloading, the script will tell you what to do next:

- **If classification dataset**: Ready to train! Proceed to training the classifier
- **If detection dataset**: Run the crop extraction script first

## Directory Structure

After downloading, you'll have:

```
data/part2/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
└── test/
```

## Manual Download (Alternative)

If you prefer to download manually from Roboflow web interface:

1. Go to: https://universe.roboflow.com/oleksandr-gorpynich/drone-detect-suvzw-gptrh
2. Click "Download Dataset"
3. Select format: "Folder" for classification or "YOLOv8" for detection
4. Extract to `data/part2/`

## Next Steps

Once your dataset is ready, you'll train the classifier:

```bash
# Create config first (will be done in next step)
python src/models/train_classifier.py \
    --data data/part2 \
    --config models/configs/classifier_config.yaml
```

## Troubleshooting

### Error: ROBOFLOW_API_KEY not found
Make sure you added the API key to `.env` file in the project root.

### Error: Project not found
Double-check the workspace and project names. You can find them in the Roboflow URL:
`https://universe.roboflow.com/{workspace}/{project}`

### Need different dataset
Edit the default values in the script or pass custom arguments:

```bash
python src/data/download_classifier_data.py \
    --workspace your-workspace \
    --project your-project \
    --version 1 \
    --format folder
```
