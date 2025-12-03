import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class AnimalDataset(Dataset):
    """Custom Dataset for loading animal images organized by class folders."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all class folders and create class mappings
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        
        # Get all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_train_transforms():
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """Test/validation transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def dataloaders(batch_size=32):
    """Create dataloaders for animal classification using ImageFolder structure."""
    dataset_dir = './animal_data'
    
    train_dataset = AnimalDataset(root_dir=dataset_dir + '/train', transform=get_train_transforms())
    val_dataset = AnimalDataset(root_dir=dataset_dir + '/val', transform=get_test_transforms())
    test_dataset = AnimalDataset(root_dir= dataset_dir + '/test', transform=get_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_dataset.classes, train_loader, val_loader, test_loader