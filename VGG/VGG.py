import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms

class VGG(nn.Module):
    def __init__(self, num_classes=80):
        super(VGG, self).__init__()

        # -------- BLOCK 1 --------
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 224 -> 112
        )

        # -------- BLOCK 2 --------
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 112 -> 56
        )

        # -------- BLOCK 3 --------
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 56 -> 28
        )

        # -------- BLOCK 4 --------
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 28 -> 14
        )

        # -------- BLOCK 5 --------
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14 -> 7
        )

        # -------- FULLY CONNECTED HEAD --------
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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

def define_params(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

def dataloaders(batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG typically uses 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using the custom AnimalDataset
    train_dataset = AnimalDataset(
        images_dir='../processed_images/train/images',
        labels_dir='../processed_images/train/labels',
        transform=transform
    )
    
    val_dataset = AnimalDataset(
        images_dir='../processed_images/val/images',
        labels_dir='../processed_images/val/labels',
        transform=transform
    )
    
    test_dataset = AnimalDataset(
        images_dir='../processed_images/test/images',
        labels_dir=None,  # Test set has no labels
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_dataset.classes, train_loader, val_loader, test_loader

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc="Training", leave=False)

    for batch in loop:
        data, target = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    print(f"Avg Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def evaluate(model, device, dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validating", leave=False)
        for batch in loop:
            data, target = batch
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f"\nAverage loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")

def display_confusion_matrix(model, device, test_loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation='vertical')
    plt.show()


if __name__ == "__main__":
    classes, train_loader, val_loader, test_loader = dataloaders(batch_size=32)
    print(f"Found {len(classes)} classes")

    model = VGG(num_classes=len(classes))
    print(model)
    criterion, optimizer = define_params(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, device, train_loader, optimizer, criterion)
        evaluate(model, device, val_loader, criterion)
    
    display_confusion_matrix(model, device, test_loader, classes)
