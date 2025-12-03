import os
import shutil
import random
from tqdm import tqdm

def create_validation_split_from_imagefolder(source_dir="animal_data", val_split=0.2, seed=42):
    """
    Create validation split from ImageFolder structured training data
    
    Args:
        source_dir: Directory containing train and test folders with class subdirectories
        val_split: Fraction of training data to use for validation (0.2 = 20%)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)
    
    train_dir = os.path.join(source_dir, "train")
    val_dir = os.path.join(source_dir, "val")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
    
    print(f"Found {len(class_dirs)} classes")
    
    total_moved = 0
    
    # Process each class
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        # Create validation class directory
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all images in this class
        image_files = [f for f in os.listdir(train_class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if len(image_files) == 0:
            print(f"No images found in {class_name}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        val_count = max(1, int(len(image_files) * val_split))  # At least 1 image for validation
        val_files = image_files[:val_count]
        
        # Move files to validation
        for img_file in val_files:
            src_path = os.path.join(train_class_dir, img_file)
            dst_path = os.path.join(val_class_dir, img_file)
            shutil.move(src_path, dst_path)
            total_moved += 1
    
    print(f"\nValidation split created successfully!")
    print(f"Total images moved to validation: {total_moved}")
    
    # Count remaining files
    train_count = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                     for d in class_dirs if os.path.exists(os.path.join(train_dir, d)))
    
    val_count = sum(len([f for f in os.listdir(os.path.join(val_dir, d)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                   for d in class_dirs if os.path.exists(os.path.join(val_dir, d)))
    
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

if __name__ == "__main__":
    create_validation_split_from_imagefolder(val_split=0.2)  # 20% for validation