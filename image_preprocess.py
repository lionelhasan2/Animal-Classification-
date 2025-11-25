import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# --- Configuration ---
DOWNLOAD_DIR = "animal_data"
PROCESSED_DIR = "processed_images"
TARGET_SIZE = (416, 416) # Common size for YOLO models
FLIP_PROBABILITY = 0.1  # 10% of images will be flipped

def process_images(input_dir, output_dir, target_size):
    """
    Performs resizing, normalization, and geometric transformations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Starting image processing from '{input_dir}'...")
    image_count = 0
    flipped_count = 0
    
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, filename)
        
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # --- Preprocessing Steps ---

        # 2. Resizing with Aspect Ratio Preservation (Padded Resize)
        h, w, _ = img.shape
        
        # Calculate scaling factor
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        # Apply padding (using a constant black color)
        padded_img = cv2.copyMakeBorder(
            resized_img, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 3. Normalization (0.0 to 1.0)
        normalized_img = padded_img.astype(np.float32) / 255.0

        # --- Save Original Image ---
        save_original = (normalized_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"orig_{filename}"), save_original)

        # --- Geometric Transformations (Data Augmentation) ---
        # 4. Horizontal Flip (Mirroring) - Only applied to 10% of images randomly
        if random.random() < FLIP_PROBABILITY:
            flipped_img = cv2.flip(normalized_img, 1)
            save_flipped = (flipped_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"flip_{filename}"), save_flipped)
            flipped_count += 1

        image_count += 1

    total_saved = image_count + flipped_count
    print(f"Finished processing. Processed {image_count} images, saved {total_saved} total ({flipped_count} flipped) to '{output_dir}'.")


if __name__ == "__main__":
    # Verify dataset directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Error: '{DOWNLOAD_DIR}' directory not found.")
        exit(1)
    
    # Process train, val, and test images
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(DOWNLOAD_DIR, split, 'images')
        output_split_dir = os.path.join(PROCESSED_DIR, split)
        
        if not os.path.exists(images_dir):
            print(f"Skipping '{split}' - directory not found at {images_dir}")
            continue
        
        image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if image_count == 0:
            print(f"Skipping '{split}' - no images found")
            continue
        
        # Check if already processed
        if os.path.exists(output_split_dir) and os.listdir(output_split_dir):
            print(f"Skipping '{split}' - already processed images found in {output_split_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        print(f"Found {image_count} images in {images_dir}")
        
        # Process images
        process_images(images_dir, output_split_dir, TARGET_SIZE) 