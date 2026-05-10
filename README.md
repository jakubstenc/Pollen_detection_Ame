# Pollen Bürker Chamber Counter

This project aims to automatically count pollen grains on a Bürker chamber using a Canon camera mounted to a microscope and a YOLOv8 Nano Instance Segmentation model.

## Directory Structure
- `data/raw_images/`: Original 5MB Canon JPGs.
- `data/tiles_640/`: Sliced 640x640 images for Roboflow labeling.
- `data/hard_negatives/`: Images containing tricky artifacts (like background debris) to help the model learn what NOT to detect.
- `data/roboflow_export/`: The .zip files and extracted datasets downloaded from Roboflow for training.
- `data/test_images/`: Separated images for final testing.
- `notebooks/`: Jupyter notebooks for experimentation.
- `src/`: Python scripts.

## Phase 1: Data Preparation
Use `src/01_prepare_tiles.py` to slice raw high-resolution images into 640x640 tiles with 20% overlap, ready for annotation in Roboflow.

## Phase 2: Model Training
Use `src/02_train_yolo.py` to train the YOLOv8 Nano instance segmentation model on the exported dataset from Roboflow. It points to `data/dataset_yolo/data.yaml` by default.
