"""
Dataset collection script for downloading drone datasets from Roboflow.
"""

import os
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int = 1,
    format: str = "yolov8",
    location: str = "./data/raw"
):
    """
    Download a dataset from Roboflow.

    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number (default: 1)
        format: Export format (default: "yolov8")
        location: Download location (default: "./data/raw")

    Example:
        download_roboflow_dataset(
            workspace="my-workspace",
            project="drone-detection",
            version=1
        )
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Please set your ROBOFLOW_API_KEY in the .env file.\n"
            "Get your API key from: https://app.roboflow.com/settings/api"
        )

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Get project
    print(f"Accessing workspace: {workspace}, project: {project}")
    project_obj = rf.workspace(workspace).project(project)

    # Download dataset
    print(f"Downloading version {version} in {format} format...")
    dataset = project_obj.version(version).download(format, location=location)

    print(f"✓ Dataset downloaded to: {location}")
    return dataset


if __name__ == "__main__":
    # Download first dataset: Drone vs Bird
    download_roboflow_dataset(
        workspace="oleksandr-gorpynich",
        project="drone-vs-bird-object-detection-2fnnk",
        version=1,
        format="yolov8",
        location="./data/raw"
    )

    # Download second dataset: Airborne Object Detection
    download_roboflow_dataset(
        workspace="oleksandr-gorpynich",
        project="airborne-object-detection-4-aod4-zaeoh",
        version=1,
        format="yolov8",
        location="./data/raw"
    )
