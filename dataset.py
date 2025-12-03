import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class AnimalDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.has_labels = labels_dir is not None and os.path.exists(labels_dir)
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label if available
        if self.has_labels:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)
            
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                    else:
                        class_id = 0
            except FileNotFoundError:
                class_id = 0  # default class if no label file
        else:
            class_id = -1  # indicate no label available
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id
    

def dataloaders(batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using the custom AnimalDataset
    train_dataset = AnimalDataset(
        images_dir='../animal_data/train/images',
        labels_dir='../animal_data/train/labels',
        transform=transform
    )
    
    val_dataset = AnimalDataset(
        images_dir='../animal_data/val/images',
        labels_dir='../animal_data/val/labels',
        transform=transform
    )
    
    test_dataset = AnimalDataset(
        images_dir='../animal_data/test/images',
        labels_dir=None,  # Test set has no labels
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_dataset.classes, train_loader, val_loader, test_loader