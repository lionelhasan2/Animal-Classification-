import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# --- Configuration ---
DATASET_DIR = "animal_data"
PROCESSED_DIR = "processed_images"
TARGET_SIZE = (416, 416)
AUGMENT_PROBABILITY = 0.1
VAL_SPLIT_RATIO = 0.2  # 20% of training data for validation

def parse_label_line(label_line):
    """Parse label line format: 'Brown Bear 212.41856 134.383104 741.982208 627.37536'""" 
    parts = label_line.strip().split()
    if len(parts) >= 5:
        # The last 4 parts are always coordinates: x_center, y_center, width, height
        # The rest is the class name (can be multi-word)
        class_name_parts = parts[:-4]
        class_name = ' '.join(class_name_parts)
        x_center = float(parts[-4])
        y_center = float(parts[-3]) 
        width = float(parts[-2])
        height = float(parts[-1])
        return class_name, x_center, y_center, width, height
    return None

def convert_to_yolo_format(class_name, x_center, y_center, width, height, img_width, img_height, class_mapping):
    """Convert absolute coordinates to YOLO normalized format"""
    if class_name not in class_mapping:
        return None
    
    class_id = class_mapping[class_name]
    norm_x_center = x_center / img_width
    norm_y_center = y_center / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}"

def flip_bounding_box_horizontal_yolo(bbox_line):
    """Flip YOLO format bounding box horizontally"""
    parts = bbox_line.strip().split()
    if len(parts) >= 5:
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = parts[2]
        width = parts[3]
        height = parts[4]
        flipped_x = 1.0 - x_center
        return f"{class_id} {flipped_x} {y_center} {width} {height}"
    return bbox_line

def flip_bounding_box_horizontal_original(bbox_line, img_width):
    """Flip original format bounding box horizontally"""
    parts = bbox_line.strip().split()
    if len(parts) >= 5:
        class_name_parts = parts[:-4]
        class_name = ' '.join(class_name_parts)
        x_center = float(parts[-4])
        y_center = parts[-3]
        width = parts[-2]
        height = parts[-1]
        flipped_x = img_width - x_center
        return f"{class_name} {flipped_x} {y_center} {width} {height}"
    return bbox_line

def process_label_file(label_path, img_width, img_height, class_mapping, flip_horizontal=False):
    """Process label file and return both YOLO and original format labels"""
    if not os.path.exists(label_path):
        return None, None
    
    yolo_labels = []
    original_labels = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parsed = parse_label_line(line)
                if parsed:
                    class_name, x_center, y_center, width, height = parsed
                    
                    if class_name not in class_mapping:
                        continue
                    
                    yolo_line = convert_to_yolo_format(
                        class_name, x_center, y_center, width, height, 
                        img_width, img_height, class_mapping
                    )
                    
                    original_line = f"{class_name} {x_center} {y_center} {width} {height}"
                    
                    if yolo_line:
                        if flip_horizontal:
                            yolo_line = flip_bounding_box_horizontal_yolo(yolo_line)
                            original_line = flip_bounding_box_horizontal_original(original_line, img_width)
                        
                        yolo_labels.append(yolo_line)
                        original_labels.append(original_line)
    except Exception as e:
        return None, None
    
    return yolo_labels, original_labels

def augment_brightness(img, factor_range=(0.7, 1.3)):
    """Adjust brightness"""
    factor = random.uniform(factor_range[0], factor_range[1])
    augmented = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return np.clip(augmented, 0, 255).astype(np.uint8)

def augment_horizontal_flip(img):
    """Flip image horizontally"""
    return cv2.flip(img, 1)

def get_class_mapping(dataset_dir, split):
    """Create class name to ID mapping from directory structure"""
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        return {}
    
    class_dirs = [d for d in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, d))]
    class_dirs.sort()
    
    return {class_name: idx for idx, class_name in enumerate(class_dirs)}

def split_train_validation(train_split_dir, val_split_ratio=0.2):
    """Split training data into train and validation sets"""
    print(f"Creating train/validation split with {val_split_ratio*100}% for validation...")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Get all training image files with their class directories
    train_files = []
    
    for class_name in os.listdir(train_split_dir):
        class_dir = os.path.join(train_split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            train_files.append((class_name, img_file))
    
    # Shuffle and split
    random.shuffle(train_files)
    split_idx = int(len(train_files) * (1 - val_split_ratio))
    
    train_subset = train_files[:split_idx]
    val_subset = train_files[split_idx:]
    
    print(f"Training subset: {len(train_subset)} images")
    print(f"Validation subset: {len(val_subset)} images")
    
    return train_subset, val_subset

def process_animal_class(class_dir, labels_dir, output_images_dir, output_yolo_labels_dir, 
                        output_original_labels_dir, class_mapping, target_size, augment=False):
    """Process all images in a single animal class directory"""
    
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    image_count = 0
    augmented_count = 0
    yolo_labels_processed = 0
    original_labels_processed = 0
    
    for filename in tqdm(image_files, desc=f"Processing {os.path.basename(class_dir)}"):
        image_path = os.path.join(class_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename) if labels_dir else None
        
        img = cv2.imread(image_path)
        if img is None:
            continue

        original_h, original_w, _ = img.shape
        
        # Preprocessing
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        normalized_img = padded_img.astype(np.float32) / 255.0
        save_img = (normalized_img * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(output_images_dir, filename), save_img)
        
        # Process labels
        if label_path and os.path.exists(label_path):
            yolo_labels, original_labels = process_label_file(
                label_path, original_w, original_h, class_mapping, flip_horizontal=False
            )
            
            base_filename = os.path.splitext(filename)[0]
            
            if yolo_labels:
                yolo_label_path = os.path.join(output_yolo_labels_dir, base_filename + '.txt')
                with open(yolo_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels) + '\n')
                yolo_labels_processed += 1
            
            if original_labels:
                original_label_path = os.path.join(output_original_labels_dir, base_filename + '.txt')
                with open(original_label_path, 'w') as f:
                    f.write('\n'.join(original_labels) + '\n')
                original_labels_processed += 1

        image_count += 1

        # Data Augmentation
        if augment and random.random() < AUGMENT_PROBABILITY and label_path and os.path.exists(label_path):
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            
            if random.random() < 0.5:
                # Brightness augmentation
                augmented_img = augment_brightness(save_img)
                aug_filename = f"aug_brightness_{base_name}{ext}"
                cv2.imwrite(os.path.join(output_images_dir, aug_filename), augmented_img)
                
                if yolo_labels and original_labels:
                    aug_base = f"aug_brightness_{base_name}"
                    
                    aug_yolo_path = os.path.join(output_yolo_labels_dir, aug_base + '.txt')
                    with open(aug_yolo_path, 'w') as f:
                        f.write('\n'.join(yolo_labels) + '\n')
                    yolo_labels_processed += 1
                    
                    aug_original_path = os.path.join(output_original_labels_dir, aug_base + '.txt')
                    with open(aug_original_path, 'w') as f:
                        f.write('\n'.join(original_labels) + '\n')
                    original_labels_processed += 1
                
                augmented_count += 1
            else:
                # Horizontal flip augmentation
                flipped_img = augment_horizontal_flip(save_img)
                flip_filename = f"aug_flip_{base_name}{ext}"
                cv2.imwrite(os.path.join(output_images_dir, flip_filename), flipped_img)
                
                flipped_yolo_labels, flipped_original_labels = process_label_file(
                    label_path, original_w, original_h, class_mapping, flip_horizontal=True
                )
                
                flip_base = f"aug_flip_{base_name}"
                
                if flipped_yolo_labels:
                    flip_yolo_path = os.path.join(output_yolo_labels_dir, flip_base + '.txt')
                    with open(flip_yolo_path, 'w') as f:
                        f.write('\n'.join(flipped_yolo_labels) + '\n')
                    yolo_labels_processed += 1
                
                if flipped_original_labels:
                    flip_original_path = os.path.join(output_original_labels_dir, flip_base + '.txt')
                    with open(flip_original_path, 'w') as f:
                        f.write('\n'.join(flipped_original_labels) + '\n')
                    original_labels_processed += 1
                
                augmented_count += 1

    return image_count, augmented_count, yolo_labels_processed, original_labels_processed

def process_split(dataset_dir, split, processed_dir, target_size):
    """Process entire split (train/test)"""
    split_dir = os.path.join(dataset_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Split directory not found: {split_dir}")
        return
    
    class_mapping = get_class_mapping(dataset_dir, split)
    if not class_mapping:
        print(f"No classes found in {split_dir}")
        return
    
    print(f"Found {len(class_mapping)} classes")
    
    # Create output directories
    output_images_dir = os.path.join(processed_dir, split, 'images')
    output_yolo_labels_dir = os.path.join(processed_dir, split, 'labels_yolo')
    output_original_labels_dir = os.path.join(processed_dir, split, 'labels_original')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_yolo_labels_dir, exist_ok=True)
    os.makedirs(output_original_labels_dir, exist_ok=True)
    
    # Save class mapping
    with open(os.path.join(processed_dir, split, 'classes.txt'), 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_id}: {class_name}\n")
    
    # Save YOLO dataset configuration
    with open(os.path.join(processed_dir, split, 'yolo_config.yaml'), 'w') as f:
        f.write(f"# YOLO dataset configuration for {split}\n")
        f.write(f"path: {os.path.abspath(processed_dir)}\n")
        f.write(f"train: {split}/images\n")
        f.write(f"val: {split}/images\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write(f"names: {list(class_mapping.keys())}\n")
    
    total_images = 0
    total_augmented = 0
    total_yolo_labels = 0
    total_original_labels = 0
    
    # Process each animal class
    for class_name in class_mapping.keys():
        class_dir = os.path.join(split_dir, class_name)
        labels_dir = os.path.join(class_dir, 'Label') if os.path.exists(os.path.join(class_dir, 'Label')) else None
        
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
        
        augment = (split == 'train')
        img_count, aug_count, yolo_count, original_count = process_animal_class(
            class_dir, labels_dir, output_images_dir, output_yolo_labels_dir, 
            output_original_labels_dir, class_mapping, target_size, augment=augment
        )
        
        total_images += img_count
        total_augmented += aug_count
        total_yolo_labels += yolo_count
        total_original_labels += original_count
    
    total_saved = total_images + total_augmented
    print(f"Finished {split}: {total_saved} images, {total_yolo_labels} YOLO labels, {total_original_labels} original labels")

def process_split_with_file_list(dataset_dir, original_split, target_split, file_list, processed_dir, target_size):
    """Process a specific list of files from the original split to create a new split"""
    
    split_dir = os.path.join(dataset_dir, original_split)
    processed_split_dir = os.path.join(processed_dir, target_split)
    
    output_images_dir = os.path.join(processed_split_dir, 'images')
    output_yolo_labels_dir = os.path.join(processed_split_dir, 'labels')
    output_original_labels_dir = os.path.join(processed_split_dir, 'original_labels')
    
    # Create directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_yolo_labels_dir, exist_ok=True)
    os.makedirs(output_original_labels_dir, exist_ok=True)
    
    # Get class mapping
    class_mapping = get_class_mapping(dataset_dir, original_split)
    
    total_images = 0
    total_yolo_labels = 0
    total_original_labels = 0
    
    for class_name, filename in tqdm(file_list, desc=f"Processing {target_split}"):
        class_dir = os.path.join(split_dir, class_name)
        labels_dir = os.path.join(class_dir, 'Label') if os.path.exists(os.path.join(class_dir, 'Label')) else None
        
        image_path = os.path.join(class_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename) if labels_dir else None
        
        if not os.path.exists(image_path):
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            continue

        original_h, original_w, _ = img.shape
        
        # Preprocessing (same as before)
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        normalized_img = padded_img.astype(np.float32) / 255.0
        save_img = (normalized_img * 255).astype(np.uint8)
        
        # Create unique filename to avoid collisions
        unique_filename = f"{class_name}_{filename}"
        cv2.imwrite(os.path.join(output_images_dir, unique_filename), save_img)
        
        # Process labels
        if label_path and os.path.exists(label_path):
            yolo_labels, original_labels = process_label_file(
                label_path, original_w, original_h, class_mapping, flip_horizontal=False
            )
            
            base_filename = os.path.splitext(unique_filename)[0]
            
            if yolo_labels:
                yolo_label_path = os.path.join(output_yolo_labels_dir, base_filename + '.txt')
                with open(yolo_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels) + '\n')
                total_yolo_labels += 1
            
            if original_labels:
                original_label_path = os.path.join(output_original_labels_dir, base_filename + '.txt')
                with open(original_label_path, 'w') as f:
                    f.write('\n'.join(original_labels) + '\n')
                total_original_labels += 1

        total_images += 1

    print(f"Finished {target_split}: {total_images} images, {total_yolo_labels} YOLO labels, {total_original_labels} original labels")

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        exit(1)
    
    # Check if train split exists
    train_path = os.path.join(DATASET_DIR, 'train')
    if os.path.exists(train_path):
        print(f"\nProcessing TRAIN split with validation split")
        
        # Create train/validation split
        train_subset, val_subset = split_train_validation(train_path, VAL_SPLIT_RATIO)
        
        # Process train subset
        print("Processing training subset...")
        process_split_with_file_list(DATASET_DIR, 'train', 'train', train_subset, PROCESSED_DIR, TARGET_SIZE)
        
        # Process validation subset  
        print("Processing validation subset...")
        process_split_with_file_list(DATASET_DIR, 'train', 'val', val_subset, PROCESSED_DIR, TARGET_SIZE)
    else:
        print("Skipping 'train' - directory not found at animal_data/train")
    
    # Process test split normally
    test_path = os.path.join(DATASET_DIR, 'test')
    if os.path.exists(test_path):
        print(f"\nProcessing TEST split")
        process_split(DATASET_DIR, 'test', PROCESSED_DIR, TARGET_SIZE)
    else:
        print("Skipping 'test' - directory not found at animal_data/test")
    
    print(f"\nProcessing complete! Check '{PROCESSED_DIR}' for results.")
    print("Your processed_images folder now contains: train/, val/, and test/ subdirectories")