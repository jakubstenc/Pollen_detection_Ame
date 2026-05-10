import os
import glob
import shutil
import datetime

def generate_report():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.join(project_root, "runs")
    docs_dir = os.path.join(project_root, "docs")
    assets_dir = os.path.join(docs_dir, "assets")
    
    os.makedirs(assets_dir, exist_ok=True)
    
    # Find the most recent training run directory
    run_folders = glob.glob(os.path.join(runs_dir, "pollen_nano_test*"))
    if not run_folders:
        print("No training runs found to generate a report.")
        return
        
    latest_run = max(run_folders, key=os.path.getmtime)
    print(f"Generating report from latest run: {os.path.basename(latest_run)}")
    
    # Count training images
    train_images_dir = os.path.join(project_root, "data", "dataset_yolo", "train", "images")
    num_images = 0
    if os.path.exists(train_images_dir):
        images = glob.glob(os.path.join(train_images_dir, "*.jpg")) + glob.glob(os.path.join(train_images_dir, "*.JPG"))
        num_images = len(images)
        
    # Files to copy to docs/assets
    target_files = {
        "results.png": "Model Training Metrics",
        "confusion_matrix.png": "Confusion Matrix",
        "train_batch0.jpg": "Ground Truth Training Data (What YOLO was taught)",
        "val_batch0_pred.jpg": "Validation Predictions (What YOLO guessed)"
    }
    
    copied_files = []
    for filename, title in target_files.items():
        src = os.path.join(latest_run, filename)
        if os.path.exists(src):
            dst = os.path.join(assets_dir, filename)
            shutil.copy2(src, dst)
            copied_files.append((filename, title))
            
    # Generate the Markdown file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = f"# YOLOv8 Pollen Detection Model Performance\n\n"
    md_content += f"**Last Updated:** {timestamp}\n\n"
    md_content += f"**Training Images Used:** {num_images}\n\n"
    
    md_content += "This page automatically tracks the performance metrics of the latest YOLOv8 Nano instance segmentation training run.\n\n"
    
    for filename, title in copied_files:
        md_content += f"## {title}\n"
        md_content += f"![{title}](assets/{filename})\n\n"
        
    # Add automated counting results section
    results_dir = os.path.join(project_root, "data", "results")
    csv_src = os.path.join(results_dir, "pollen_counts.csv")
    csv_filename = "pollen_counts.csv"
    
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, os.path.join(assets_dir, csv_filename))
        md_content += f"\n## Automated Pollen Counting Results\n\n"
        md_content += f"The Bürker grid has been automatically aligned and pollen counted according to the standard **Hemocytometer L-Rule (North/West)**.\n\n"
        md_content += f"> [!NOTE]\n> **Counting Protocol**:\n> - **Counted (Green)**: Pollen inside the square OR touching the **Top (North)** or **Left (West)** boundaries.\n> - **Ignored (Red)**: Pollen touching the **Bottom (South)** or **Right (East)** boundaries.\n\n"
        md_content += f"**📥 [Download Raw Data CSV (pollen_counts.csv)](assets/{csv_filename})**\n\n"
        
        # Read CSV and build a Markdown table
        import csv
        try:
            with open(csv_src, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
                
            if headers and rows:
                md_content += "### 9-Square Statistical Breakdown\n\n"
                md_content += "|" + "|".join(headers) + "|\n"
                md_content += "|" + "|".join(["---"] * len(headers)) + "|\n"
                for row in rows:
                    md_content += "|" + "|".join(row) + "|\n"
                md_content += "\n"
        except Exception as e:
            print(f"Failed to generate markdown table from CSV: {e}")
            
        md_content += f"## 🎥 Real-Time Live Camera Tracking\n\n"
        md_content += f"You can bypass static images and run the system directly on a live camera stream! It runs inference directly on the video frames and dynamically snaps the mathematical grid into place.\n\n"
        md_content += f"To run the live stream from your terminal:\n"
        md_content += f"```bash\npython src/live_counter.py --camera 0 --model runs/pollen_nano_test-3/weights/best.pt\n```\n"
        md_content += f"*(Change `--camera 0` to a specific index or RTSP stream URL depending on your microscope setup).* \n\n"
        md_content += f"**Live Features**:\n- **`q`** : Quit the live stream.\n- **`g`** : Force an instant re-calculation of the mathematical grid.\n\n"
        
        # Add visual results gallery
        vis_images = glob.glob(os.path.join(results_dir, "visualized_*.JPG")) + glob.glob(os.path.join(results_dir, "visualized_*.jpg"))
        for vis_src in vis_images:
            vis_filename = os.path.basename(vis_src)
            shutil.copy2(vis_src, os.path.join(assets_dir, vis_filename))
            md_content += f"### {vis_filename}\n"
            md_content += f"![{vis_filename}](assets/{vis_filename})\n\n"
            
    md_path = os.path.join(docs_dir, "index.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"Successfully generated documentation at {md_path}")

if __name__ == "__main__":
    generate_report()
