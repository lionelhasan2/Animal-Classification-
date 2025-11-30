import os
import shutil
from tqdm import tqdm 
'''
    Organizes images into folder structure for YOLO to use
    image organization file built by ChatGPT
'''

BASE_DIR = r"C:\Users\Eric\Coding\Animal-Classification-\processed_images"
OUTPUT_DIR = r"C:\Users\Eric\Coding\Animal-Classification-\classification_data"
CLASS_NAMES = ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra']

def process_split(split_name):
    print(f"\nProcessing {split_name} set...")
    
    labels_dir = os.path.join(BASE_DIR, split_name, "labels")
    images_dir = os.path.join(BASE_DIR, split_name, "images")
    
    output_split_dir = os.path.join(OUTPUT_DIR, split_name)

    if not os.path.exists(labels_dir):
        print(f"Skipping {split_name} (Folder not found: {labels_dir})")
        return

    # Create the output directories for all classes ahead of time
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(output_split_dir, name), exist_ok=True)

    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    
    count = 0
    for label_file in label_files:
        # 1. Read the Class ID from the file
        label_path = os.path.join(labels_dir, label_file)
        
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            if not lines:
                continue # Empty file
            
            # Grab the first number in the file (the class ID)
            class_id = int(lines[0].split()[0])
            
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                print(f"Warning: Found invalid class ID {class_id} in {label_file}")
                continue

            class_name = CLASS_NAMES[class_id]
            
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue

        # 2. Find the matching image
        base_name = os.path.splitext(label_file)[0]
        
        image_found = False
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            img_name = base_name + ext
            src_img_path = os.path.join(images_dir, img_name)
            
            if os.path.exists(src_img_path):
                # 3. Copy image to the new folder
                dst_img_path = os.path.join(output_split_dir, class_name, img_name)
                shutil.copy2(src_img_path, dst_img_path)
                image_found = True
                count += 1
                break
        
        if not image_found:
            pass

    print(f"Successfully organized {count} images for {split_name}.")

if __name__ == "__main__":
    process_split("train")
    process_split("test") 
    print("\n--- Finished ---")

