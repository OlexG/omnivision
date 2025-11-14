# Web Test Images for Drone Classification

This directory is for testing the two-stage pipeline with diverse drone types.

## What to Add

Download and save drone images here (`.jpg` or `.png` format) to test classification on:

### Drone Types to Test:

1. **Fixed wing** - Military UAVs like MQ-9 Reaper, Predator
2. **VTOL** - Vertical takeoff/landing hybrid drones
3. **FPV** - Racing drones, small high-speed quads
4. **Hexacopter** - 6-rotor drones
5. **Octocopter** - 8-rotor heavy lift drones
6. **DJI Mavic** - Foldable consumer drones
7. **DJI Phantom** - Popular consumer quadcopters
8. **DJI FPV** - DJI's FPV racing drone
9. **Cinewhoop** - Tiny whoop style FPV drones
10. **Pluto Mini** - Small toy drones

## Where to Find Images

### Free Sources:
- **Unsplash**: https://unsplash.com/s/photos/drone
- **Pexels**: https://www.pexels.com/search/drone/
- **Wikimedia Commons**: https://commons.wikimedia.org/wiki/Category:Drones
- **Google Images**: Use "Tools > Usage rights > Creative Commons licenses"

### Search Terms:
```
"MQ-9 Reaper drone"
"Predator military drone"
"VTOL drone hybrid"
"FPV racing drone quadcopter"
"hexacopter 6 rotor"
"octocopter heavy lift"
"DJI Mavic Pro"
"DJI Phantom 4"
"cinewhoop tiny whoop"
"fixed wing UAV"
```

## Usage

Once you've added images here:

1. Open `notebooks/demo_inference_full.ipynb`
2. Run all cells
3. The notebook will automatically find and process images in this directory
4. View color-coded classifications for each drone type!

## Example Results

The pipeline will:
- ✅ Detect drones with YOLOv8 (green solid box)
- ❌ Detect non-drones like birds/planes (red dashed box)
- 🎨 Classify drone types with color-coded labels
- 📊 Show top-3 predictions with confidence scores

## Notes

- **"not-drone" detections**: Will show with red dashed boxes and won't be classified
- **Only "drone" detections**: Will be classified into the 10 drone types
- **Diverse testing**: Try images with different angles, lighting, and backgrounds
- **Performance**: Classification works best on clear, well-lit drone images

Enjoy testing! 🚁
