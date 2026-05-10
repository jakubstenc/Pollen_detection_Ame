import os
from ultralytics import YOLO

def main():
    # Define absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_yaml_path = os.path.join(project_root, "data", "dataset_yolo", "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found.")
        print("Please extract your Roboflow zip into data/dataset_yolo/ first.")
        return

    print("Initializing YOLOv8 Nano Instance Segmentation model...")
    # Load the pre-trained Nano segmentation model
    model = YOLO("yolov8n-seg.pt")
    
    print(f"Starting training on dataset defined in: {data_yaml_path}")
    
    # Train the model
    # We use a low number of epochs (e.g. 20) for local testing purposes.
    # When training properly on a GPU/cluster, increase epochs to 100-300.
    results = model.train(
        data=data_yaml_path,
        epochs=20,           # Set low for quick local testing
        imgsz=640,           # The size we sliced our tiles to
        batch=4,             # Keep batch size small to save RAM/VRAM
        project=os.path.join(project_root, "runs"),
        name="pollen_nano_test",
        patience=5           # Early stopping if no improvement
    )
    
    print("\nTraining completed!")
    print(f"Your best model weights are saved at: {os.path.join(project_root, 'runs', 'pollen_nano_test', 'weights', 'best.pt')}")

    # Generate the automated documentation report
    print("\nGenerating performance report...")
    try:
        from generate_report import generate_report
        generate_report()
    except Exception as e:
        print(f"Failed to generate report: {e}")

if __name__ == "__main__":
    main()
