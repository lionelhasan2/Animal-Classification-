import os
import shutil
import random

def split_dataset(dataset_dir, train_ratio, seed):
    """Splits a classification dataset into separate 'train' and 'test' directories."""
    
    try:
        print("Starting split...")

        random.seed(seed)

        # Define paths for the new train and test directories
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_name)

            # Get a list of all image files in the current class folder
            images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if len(images) == 0:
                continue

            # Shuffle the list of images to ensure random split
            random.shuffle(images)

            # Split images to 80% train 20% test per animal folder
            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Move images
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_class_dir, img)
                shutil.move(src, dst)

            for img in test_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(test_class_dir, img)
                shutil.move(src, dst)
                            
            # Remove empty class name folder
            if not os.listdir(class_path):
                os.rmdir(class_path)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "Animals-10") # Animals-10 is the dataset folder name
    train_ratio = 0.8 # 80% train, 20% test
    random_seed = 42 # for reproducible splits
    split_dataset(dataset_dir,train_ratio,random_seed)
    print("\nDataset split complete.")