import os
import cv2
import glob

def slice_image(image_path, output_dir, tile_size=640, overlap_pct=0.2):
    """
    Slices a large image into tiles of `tile_size` x `tile_size` with a specified overlap.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory where tiles will be saved.
        tile_size (int): The size of the square tile.
        overlap_pct (float): The percentage of overlap between consecutive tiles.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    img_height, img_width = img.shape[:2]
    
    # Calculate stride based on overlap
    stride = int(tile_size * (1 - overlap_pct))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract original filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Processing '{base_name}' ({img_width}x{img_height}) ...")
    
    # Track row and column indices
    row = 0
    for y in range(0, img_height, stride):
        col = 0
        y_end = y + tile_size
        
        # Adjust if the tile goes beyond the image bottom edge
        if y_end > img_height:
            y_start = max(0, img_height - tile_size)
            y_end = img_height
        else:
            y_start = y

        for x in range(0, img_width, stride):
            x_end = x + tile_size
            
            # Adjust if the tile goes beyond the image right edge
            if x_end > img_width:
                x_start = max(0, img_width - tile_size)
                x_end = img_width
            else:
                x_start = x

            # Extract the tile
            tile = img[y_start:y_end, x_start:x_end]
            
            # Save the tile
            tile_filename = f"{base_name}_row{row:03d}_col{col:03d}.jpg"
            tile_path = os.path.join(output_dir, tile_filename)
            cv2.imwrite(tile_path, tile)
            
            col += 1
            
            # If we've hit the right edge, break out of this row
            if x_end == img_width:
                break
        
        row += 1
        
        # If we've hit the bottom edge, break completely
        if y_end == img_height:
            break

    print(f"Finished processing '{base_name}'. Created {row * col} tiles.\n")


def process_directory(input_dir, output_dir, tile_size=640, overlap_pct=0.2):
    """
    Processes all JPG images in a directory.
    """
    search_pattern = os.path.join(input_dir, "*.jpg")
    # You can also add *.jpeg or *.png if needed
    image_paths = glob.glob(search_pattern) + glob.glob(os.path.join(input_dir, "*.JPG"))
    
    if not image_paths:
        print(f"No JPG images found in '{input_dir}'. Please place your raw images there.")
        return
        
    for path in image_paths:
        slice_image(path, output_dir, tile_size, overlap_pct)

if __name__ == "__main__":
    # Define directories relative to this script or as absolute paths
    # Assuming this is run from the project root or src directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    INPUT_DIR = os.path.join(project_root, "data", "raw_images")
    OUTPUT_DIR = os.path.join(project_root, "data", "tiles_640")
    
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 30)
    
    process_directory(INPUT_DIR, OUTPUT_DIR, tile_size=640, overlap_pct=0.2)
