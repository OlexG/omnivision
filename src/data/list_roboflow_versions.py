"""
List available versions for a Roboflow project.
Useful for finding the correct version number before downloading.

Usage:
    python src/data/list_roboflow_versions.py \
        --workspace oleksandr-gorpynich \
        --project drone-detect-suvzw-gptrh
"""

import os
import argparse
from roboflow import Roboflow
from dotenv import load_dotenv


def list_project_versions(workspace: str, project: str):
    """List all available versions for a Roboflow project."""

    # Load API key
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in .env file. "
            "Please add it: ROBOFLOW_API_KEY=your_api_key_here\n"
            "Get your API key from: https://app.roboflow.com/settings/api"
        )

    print(f"\n{'='*60}")
    print("ROBOFLOW PROJECT VERSIONS")
    print("="*60)
    print(f"\nWorkspace: {workspace}")
    print(f"Project: {project}")

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Get project
    print(f"\nAccessing project...")
    project_obj = rf.workspace(workspace).project(project)

    # Get project info
    print(f"\nProject Name: {project_obj.name}")
    print(f"Project Type: {project_obj.type}")

    # List versions
    print(f"\nAvailable Versions:")

    # Try to access versions attribute
    if hasattr(project_obj, 'versions'):
        versions = project_obj.versions()
        if versions:
            for v in versions:
                print(f"  Version {v['id']}: {v.get('name', 'No name')}")
        else:
            print("  No versions found")
    else:
        # Try accessing version  directly - start from 1 and try incrementally
        print("  Checking available versions...")
        found_versions = []

        for i in range(1, 20):  # Check versions 1-19
            try:
                v = project_obj.version(i)
                found_versions.append(i)
                print(f"  ✓ Version {i} found")
            except:
                continue

        if not found_versions:
            print("  ⚠ Could not automatically detect versions")
            print(f"\n  Please check your project page:")
            print(f"  https://universe.roboflow.com/{workspace}/{project}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="List available versions for a Roboflow project"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="oleksandr-gorpynich",
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="drone-detect-suvzw-gptrh",
        help="Roboflow project name"
    )

    args = parser.parse_args()

    list_project_versions(args.workspace, args.project)


if __name__ == "__main__":
    main()
