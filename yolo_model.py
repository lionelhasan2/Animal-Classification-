import os
from ultralytics import YOLO
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
## Pre-Run Checklist for Yolo model:
1. Run the image_organization.py to generate the proper dataset needed for yolo
Folder Structure:
    classification_data
        -> test
            -> class1
            -> class2
            -> ...
        -> train
            -> class1
            -> class2
            -> ...
'''

def init(model_name):
    '''Create Yolo model & determine device to use'''
    model = YOLO(model_name) 
    if torch.cuda.is_available(): # gpu
        device = 'cuda'
    elif torch.backends.mps.is_available(): # mac
        device = 'mps'
    else:
        device = torch.device("cpu")
    return model, device

def train(model, device, dataset_path, epoch_size, img_size, batch_size):
    ''' Start training process '''
    results = model.train(
        data=dataset_path,
        epochs=epoch_size,
        imgsz=img_size,
        batch=batch_size,
        name='yolo_v11_run',
        project='yolo_model', # Name of the folder where results are saved
        patience=50, # Early stopping if no improvement
        exist_ok=True, # Overwrite existing run folder
        save=False, # Don't keep the weights
        device=device,
        verbose=False, # 
        plots=True, # automatically saves confusion matrix, PR curves, and F1 curves
        seed = 42 # Start randomness in the exact same way every time we train model
    ) 

def evaluate(model, device, dataset_path):
    ''' Start evaluating process '''

    y_true = []
    y_pred = []
    image_paths = []

    class_names = sorted(os.listdir(dataset_path))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)} # map class names to idx

    for class_name in class_names: # map the images in the folder to the folder name (class id)
        class_path = os.path.join(dataset_path, class_name)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                image_paths.append((img_path, class_name))

    for img_path, true_class in tqdm(image_paths, desc="Evaluation"): # make a prediction per image
        result = model.predict(source=img_path, conf=0.5, verbose=False)[0]

        pred_class_idx = int(result.probs.top1)
        pred_class_name = model.names[pred_class_idx]

        y_true.append(true_class)
        y_pred.append(pred_class_name)

    y_true_idx = [class_to_idx[label] for label in y_true]
    y_pred_idx = [class_to_idx[label] for label in y_pred]

    # ---- METRICS ----
    # Use average='macro' here for equal weighting per class
    # Prevents classes with more images from skewing our metrics 
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision = precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    recall = recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    f1 = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")

    # Show confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.show()

def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(ROOT_DIR, "classification_data")
    TEST_DIR = os.path.join(BASE_DIR, "test")

    EPOCHS = 10
    IMG_SIZE = 416
    BATCH_SIZE = 8
    MODEL_NAME = 'yolo11n-cls.pt'

    print('--- Initialization ---')
    model, device = init(MODEL_NAME)
    print(f'Using model {MODEL_NAME} & device {device}')

    print('\n--- Training Model ---')
    train(model, device, dataset_path=BASE_DIR, epoch_size=EPOCHS, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    print('\n--- Evaluating Model ---')
    evaluate(model, device, dataset_path=TEST_DIR)

if __name__ == '__main__':
    main()