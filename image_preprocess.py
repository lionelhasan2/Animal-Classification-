import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# --- Configuration ---
DOWNLOAD_DIR = "animal_data"
PROCESSED_DIR = "processed_images"
TARGET_SIZE = (416, 416)
AUGMENT_PROBABILITY = 0.1

def extract_bounding_box(label_path, output_path):
    """Extract only the bounding box from polygon labels"""
    with open(label_path, 'r') as f:
        content = f.read().strip()
    
    if content:
        # Split and take only first 5 elements (class + bbox)
        parts = content.split()
        if len(parts) >= 5:
            bbox_only = ' '.join(parts[:5])  # class x_center y_center width height
            
            with open(output_path, 'w') as f:
                f.write(bbox_only + '\n')
        else:
            # Copy as-is if format is unexpected
            with open(output_path, 'w') as f:
                f.write(content)

def flip_bounding_box_horizontal(bbox_line):
    """Flip bounding box horizontally (x_center = 1.0 - x_center)"""
    parts = bbox_line.strip().split()
    if len(parts) >= 5:
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = parts[2]
        width = parts[3]
        height = parts[4]
        
        # Flip x_center: new_x = 1.0 - old_x
        flipped_x = 1.0 - x_center
        
        return f"{class_id} {flipped_x} {y_center} {width} {height}\n"
    return bbox_line

def copy_label_file(src_labels_dir, dst_labels_dir, original_filename, new_filename, flip_horizontal=False):
    """Copy and process label file with new filename to match processed image."""
    if not src_labels_dir or not os.path.exists(src_labels_dir):
        return False
    
    # Convert image filename to label filename
    original_label = os.path.splitext(original_filename)[0] + '.txt'
    new_label = os.path.splitext(new_filename)[0] + '.txt'
    
    src_label_path = os.path.join(src_labels_dir, original_label)
    
    if os.path.exists(src_label_path):
        os.makedirs(dst_labels_dir, exist_ok=True)
        dst_label_path = os.path.join(dst_labels_dir, new_label)
        
        # Read original label and extract bounding box
        with open(src_label_path, 'r') as f:
            content = f.read().strip()
        
        if content:
            # Extract bounding box only
            parts = content.split()
            if len(parts) >= 5:
                bbox_only = ' '.join(parts[:5])  # class x_center y_center width height
                
                # Apply horizontal flip if needed
                if flip_horizontal:
                    bbox_only = flip_bounding_box_horizontal(bbox_only).strip()
                
                with open(dst_label_path, 'w') as f:
                    f.write(bbox_only + '\n')
            else:
                # Copy as-is if format is unexpected
                with open(dst_label_path, 'w') as f:
                    f.write(content)
        
        return True
    return False

def augment_brightness(img, factor_range=(0.7, 1.3)):
    """Adjust brightness without changing object positions."""
    factor = random.uniform(factor_range[0], factor_range[1])
    augmented = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return np.clip(augmented, 0, 255).astype(np.uint8)

def augment_horizontal_flip(img):
    """Flip image horizontally."""
    return cv2.flip(img, 1)

def process_images(input_dir, output_dir, labels_dir, target_size, augment=False):
    """
    Process images and copy corresponding labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create labels directory in processed folder
    output_labels_dir = os.path.join(os.path.dirname(output_dir), 'labels')
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Starting image processing from '{input_dir}'...")
    image_count = 0
    augmented_count = 0
    labels_copied = 0
    
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, filename)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # --- Preprocessing Steps ---
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
        
        # Apply padding
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Normalize
        normalized_img = padded_img.astype(np.float32) / 255.0
        save_original = (normalized_img * 255).astype(np.uint8)
        
        # Save original with same filename (no prefix)
        cv2.imwrite(os.path.join(output_dir, filename), save_original)
        
        # Copy original label file (extract bounding box only)
        if copy_label_file(labels_dir, output_labels_dir, filename, filename, flip_horizontal=False):
            labels_copied += 1

        # --- Data Augmentation (Training Only) ---
        if augment and random.random() < AUGMENT_PROBABILITY:
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            
            # Brightness augmentation
            if random.random() < 0.5:  # 50% chance for brightness
                augmented_img = augment_brightness(save_original)
                aug_filename = f"aug_brightness_{base_name}{ext}"
                cv2.imwrite(os.path.join(output_dir, aug_filename), augmented_img)
                
                # Copy label file for brightness augmentation (no coordinate changes)
                if copy_label_file(labels_dir, output_labels_dir, filename, aug_filename, flip_horizontal=False):
                    labels_copied += 1
                
                augmented_count += 1
            
            # Horizontal flip augmentation
            else:  # 50% chance for horizontal flip
                flipped_img = augment_horizontal_flip(save_original)
                flip_filename = f"aug_flip_{base_name}{ext}"
                cv2.imwrite(os.path.join(output_dir, flip_filename), flipped_img)
                
                # Copy and modify label file for horizontal flip
                if copy_label_file(labels_dir, output_labels_dir, filename, flip_filename, flip_horizontal=True):
                    labels_copied += 1
                
                augmented_count += 1

        image_count += 1

    total_saved = image_count + augmented_count
    print(f"Finished processing. Processed {image_count} images, saved {total_saved} total ({augmented_count} augmented) to '{output_dir}'.")
    print(f"Copied {labels_copied} label files to '{output_labels_dir}'.")

if __name__ == "__main__":
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Error: '{DOWNLOAD_DIR}' directory not found.")
        exit(1)
    
    # Define splits with labels
    splits = {
        'train': {
            'images': os.path.join(DOWNLOAD_DIR, 'train', 'images'),
            'labels': os.path.join(DOWNLOAD_DIR, 'train', 'labels')
        },
        'val': {
            'images': os.path.join(DOWNLOAD_DIR, 'val', 'images'),
            'labels': os.path.join(DOWNLOAD_DIR, 'val', 'labels')
        },
        'test': {
            'images': os.path.join(DOWNLOAD_DIR, 'test'),
            'labels': None  # Test usually has no labels
        }
    }
    
    for split, paths in splits.items():
        images_dir = paths['images']
        labels_dir = paths['labels']
        output_split_dir = os.path.join(PROCESSED_DIR, split, 'images')
        
        if not os.path.exists(images_dir):
            print(f"Skipping '{split}' - directory not found at {images_dir}")
            continue
        
        image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if image_count == 0:
            print(f"Skipping '{split}' - no images found")
            continue
        
        if os.path.exists(output_split_dir) and os.listdir(output_split_dir):
            print(f"Skipping '{split}' - already processed images found in {output_split_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        print(f"Found {image_count} images in {images_dir}")
        
        # Only augment training images
        augment = (split == 'train')
        process_images(images_dir, output_split_dir, labels_dir, TARGET_SIZE, augment=augment)