# Pollen Bürker Chamber Counter

This project aims to automatically count pollen grains on a Bürker chamber using a Canon camera mounted to a microscope and a YOLOv8 Nano Instance Segmentation model.

## Directory Structure
- `data/raw_images/`: Original 5MB Canon JPGs.
- `data/tiles_640/`: Sliced 640x640 images for Roboflow labeling.
- `data/test_images/`: Separated images for final testing.
- `notebooks/`: Jupyter notebooks for experimentation.
- `src/`: Python scripts.

## Phase 1
Use `src/01_prepare_tiles.py` to slice raw high-resolution images into 640x640 tiles with 20% overlap, ready for annotation in Roboflow.
