import os
import shutil
import random

"""
THIS FILE WAS MADE USING CHATGPT

THIS FILE SPLITS THE DATASET INTO TRAIN & TEST DATASET FOLDERS
"""

# =======================
# CONFIG
# =======================
# ⚠️ MAKE SURE THIS PATH IS CORRECT FOR YOUR SYSTEM ⚠️
DATASET_DIR = r"C:\Users\Eric\Coding\Animal-Classification-\Animals-10" 
TRAIN_RATIO = 0.8                      # 80% train, 20% test
SEED = 42                              # for reproducible splits

# =======================
# SETUP
# =======================
# Set the seed for reproducibility
random.seed(SEED)

# Define paths for the new train and test directories
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

# Create the main train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print(f"Starting split for dataset at: {DATASET_DIR}")
print(f"Split ratio: {TRAIN_RATIO*100}% Train, {(1-TRAIN_RATIO)*100}% Test")
print("-" * 30)

# =======================
# PROCESS EACH CLASS
# =======================
# Iterate through all items (folders) inside the main dataset directory
for class_name in os.listdir(DATASET_DIR):

    class_path = os.path.join(DATASET_DIR, class_name)

    # Skip the new 'train'/'test' folders if the script is re-run
    if class_name.lower() in ["train", "test"]:
        continue

    # Only process actual class folders (not loose files)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    # Get a list of all image files in the current class folder
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]

    if len(images) == 0:
        print(f"  ⚠️ Skipping empty folder or no image files found: {class_name}")
        continue

    # Shuffle the list of images to ensure random split
    random.shuffle(images)

    # Calculate the split point
    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create the class subfolders inside the new train/test directories
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        # Use shutil.move() to move the file from source to destination
        shutil.move(src, dst)

    # Move test images
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.move(src, dst)
        
    print(f"  Train: {len(train_images)} | Test: {len(test_images)}")
    
    # Clean up the original class folder if it's now empty
    if not os.listdir(class_path):
        os.rmdir(class_path)
        print(f"  🗑️ Original folder removed: {class_name}")

print("-" * 30)
print("\n✅ Dataset split complete!")