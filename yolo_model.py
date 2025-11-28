import os
from ultralytics import YOLO

class YoloModelClass:
    def __init__(self, model_name='yolo11n.pt', device='cuda'): #mps for mac, cuda for gpu
        self.device = device
        self.model = YOLO(model_name)

    def train_model(self, data_path, epochs=50, img_size=640, batch_size=8):
        print(f"Starting training...")

        results = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='yolo11_training',
            project='runs/detect', # Name of the folder where results are saved
            patience=50, # Early stopping if no improvement
            exist_ok=True, # Overwrite existing run folder
            save=True,
            device=self.device,
            verbose=True,
            plots=True,
        )

        self.best_model_path = self.model.trainer.best
        print(f"\nTraining complete! Best weights saved at: {self.best_model_path}")

        return self.best_model_path

    def evaluate_model(self, data_path, best_weight_model_path):
        eval_model = YOLO(best_weight_model_path)

        metrics = eval_model.val(
            data=data_path,
            device=self.device
        )
        
        print("\n✅ Validation Results:")
        print(f"Precision (P):     {metrics.box.mp:.4f}") # Mean Precision
        print(f"Recall (R):        {metrics.box.mr:.4f}") # Mean Recall
        print(f"mAP@0.5:           {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95:      {metrics.box.map:.4f}")
            
        return metrics