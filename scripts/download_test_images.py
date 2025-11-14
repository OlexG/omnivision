"""
Download diverse test drone images for classification testing.
These images cover different drone types: fixed-wing, VTOL, military, FPV, etc.

Usage:
    python scripts/download_test_images.py
"""

import urllib.request
import ssl
from pathlib import Path

# Create directory
output_dir = Path("data/web_test_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Diverse drone image URLs
# Using Wikimedia Commons direct links (working URLs)
test_images = {
    # Fixed wing military drone (MQ-9 Reaper)
    "fixed_wing_military.jpg": "https://upload.wikimedia.org/wikipedia/commons/d/d3/MQ-9_Reaper_in_flight_%282007%29.jpg",

    # Military Predator drone
    "predator_drone.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/c7/MQ-1_Predator_unmanned_aircraft.jpg",

    # DJI Phantom quadcopter
    "dji_phantom.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/67/DJI_Phantom_3_Professional.jpg",

    # DJI Mavic
    "dji_mavic.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/4e/DJI_Mavic_Pro.jpg",

    # Racing/FPV drone
    "fpv_racing.jpg": "https://upload.wikimedia.org/wikipedia/commons/f/f0/Racing_Drone.jpg",
}

# Disable SSL verification (for some sites)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

print("Downloading test drone images...")
print(f"Output directory: {output_dir.absolute()}\n")

downloaded = 0
failed = 0

for filename, url in test_images.items():
    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        print(f"⏭  Skipping {filename} (already exists)")
        downloaded += 1
        continue

    try:
        print(f"📥 Downloading {filename}...")
        print(f"   URL: {url}")

        # Download with custom headers
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

        with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())

        print(f"   ✓ Saved to {filename}\n")
        downloaded += 1

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}\n")
        failed += 1

print("="*60)
print(f"DOWNLOAD COMPLETE")
print("="*60)
print(f"Downloaded: {downloaded}")
print(f"Failed: {failed}")
print(f"Total: {len(test_images)}")
print(f"\n✓ Images saved to: {output_dir.absolute()}")
print("\nNext steps:")
print("1. Run the demo_inference_full.ipynb notebook")
print("2. Point it to this directory to test classification on diverse drones")
