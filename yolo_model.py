import os
from ultralytics import YOLO
import torch

'''
## Pre-Run Checklist for Yolo model:

1. Rename label folders
    - Yolo is hardcoded to look for folder specifically named 'labels' and 'images'
    - Go to processed_images/train/ and rename labels_yolo to labels
    - Go to processed_images/test/ and rename labels_yolo to labels

2. Update yolo_configuration.yaml file to validate with a different image dataset
    - Go to processed_images/train/yolo_config.yaml
    - replace, val: train/images, with, val: test/images
'''

#Global vars
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CONFIG_PATH = os.path.join(ROOT_DIR, "processed_images", "train", "yolo_config.yaml")

PROJECT_NAME = 'yolo_model'
RUN_NAME = 'yolo_v11_run'
MODEL_NAME = 'yolo11n.pt'

EPOCHS = 3
IMG_SIZE = 640
BATCH_SIZE = 8

def main():
    print("\n--- Init ---")
    model = YOLO(MODEL_NAME) 
    if torch.cuda.is_available(): # gpu
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available(): # mac
        DEVICE = 'mps'
    else:
        DEVICE = torch.device("cpu")
    print(f"Using {MODEL_NAME} and {DEVICE}.")
    
    print("\n--- Starting Training ---")
    results = model.train(
        data=TRAIN_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        project=PROJECT_NAME, # Name of the folder where results are saved
        patience=50, # Early stopping if no improvement
        exist_ok=True, # Overwrite existing run folder
        save=True,
        device=DEVICE,
        verbose=False, # 
        plots=True, # automatically saves confusion matrix, PR curves, and F1 curves
        seed = 42 # Start randomness in the exact same way every time we train model
    ) 

    print("\n--- Starting Evaluation ---")
    metrics = model.val(data=TRAIN_CONFIG_PATH, 
                        plots=True, 
                        device=DEVICE, 
                        verbose=False)
    
    # Extract Metrics
    precision = metrics.box.mp   # mean precision
    recall = metrics.box.mr      # mean recall
    map50 = metrics.box.map50    # mAP @ IoU 0.5
    map_overall = metrics.box.map # mAP @ IoU 0.5:0.95
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Add a small epsilon (1e-7) to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    print("\n--- YOLO MODEL RESULTS --- ")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1_score:.4f}")
    print(f"mAP@0.50:          {map50:.4f} (Accuracy Proxy)")
    print(f"mAP@0.50:0.95:     {map_overall:.4f}")
    
    save_dir = results.save_dir
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')

    print(f"Confusion Matrix saved to: {cm_path}")
    print(f"Full results saved to:     {save_dir}")

if __name__ == '__main__':
    main()