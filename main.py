from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.nn as nn
import torch
from dataset import dataloaders, get_test_transforms
from VGG.VGG import VGG
from AlexNet.AlexNet import AlexNet
import random
from PIL import Image
import random

def define_params(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


def train(model, device, train_loader, optimizer, criterion, history):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
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

    history["train_loss"].append(avg_loss)
    history["train_acc"].append(accuracy)
    print(f"Avg Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def evaluate(model, device, dataloader, criterion, history):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
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

    history["train_loss"].append(avg_loss)
    history["train_acc"].append(accuracy)
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


def Train_and_Validate_VGG(num_epochs, device, classes, train_loader, val_loader, test_loader):
    """Train and Validate VGG model."""
    print("Validating VGG model...")
    model = VGG(num_classes=len(classes))
    print(model)
    criterion, optimizer = define_params(model)

    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        history = train(model, device, train_loader, optimizer, criterion, history)
        history = evaluate(model, device, val_loader, criterion, history)

    # Save model weights
    save_dir = "./VGG"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "VGG_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"VGG Model weights saved to: {save_path}")

    display_confusion_matrix(model, device, test_loader, classes)


def Train_and_Validate_AlexNet(num_epochs, device, classes, train_loader, val_loader, test_loader):
    """Train and Validate AlexNet model."""
    print("Validating AlexNet model...")
    model = AlexNet(num_classes=len(classes))
    print(model)
    criterion, optimizer = define_params(model)

    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        history = train(model, device, train_loader, optimizer, criterion, history)
        history = evaluate(model, device, val_loader, criterion, history)
    
    # Save model weights
    save_dir = "./AlexNet"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "alexnet_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"AlexNet Model weights saved to: {save_path}")

    display_confusion_matrix(model, device, test_loader, classes)
    return history


def testModel(model_name, classes, device, test_data_path='./animal_data/test'):
    """Test a trained model on a random image from the test folder."""
    # --- Load the trained model ---
    if model_name == "VGG":
        model = VGG(num_classes=len(classes))
        model.load_state_dict(torch.load('./VGG/VGG_weights.pth', map_location=device))
    elif model_name == "AlexNet":
        model = AlexNet(num_classes=len(classes))
        model.load_state_dict(torch.load('./AlexNet/alexnet_weights.pth', map_location=device))
    else:
        raise ValueError("Model must be 'VGG' or 'AlexNet'")

    model.to(device)
    model.eval()

    # Pick a random class folder
    class_dir = random.choice(os.listdir(test_data_path))
    class_path = os.path.join(test_data_path, class_dir)

    # Pick a random image inside that folder
    img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)
    print(f"\nTesting image: {img_path}")

    # Define preprocessing
    transform = get_test_transforms()

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0).to(device)   # add batch dimension

    # Run model
    with torch.no_grad():
        outputs = model(img_transformed)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = classes[predicted.item()]
    true_class = class_dir

    # Show result
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} | True: {true_class}")
    plt.axis("off")
    plt.show()

    print(f"Prediction: {predicted_class}")
    print(f"True Label: {true_class}")

def compareModels(VGG_history, AlexNet_history, num_epochs):
    """Compare training and validation accuracy of VGG and AlexNet models."""
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, VGG_history['train_acc'], label='VGG Train Acc')
    plt.plot(epochs, AlexNet_history['train_acc'], label='AlexNet Train Acc')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, VGG_history['val_acc'], label='VGG Val Acc')
    plt.plot(epochs, AlexNet_history['val_acc'], label='AlexNet Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get data loaders
    classes, train_loader, val_loader, test_loader = dataloaders(batch_size=32)
    print(f"Found {len(classes)} classes")

    # Set device and number of epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5

    VGG_history = Train_and_Validate_VGG(num_epochs, device, classes, train_loader, val_loader, test_loader)
    AlexNet_history = Train_and_Validate_AlexNet(num_epochs, device, classes, train_loader, val_loader, test_loader)
    compareModels(VGG_history, AlexNet_history, num_epochs)

    testModel("VGG", classes, device)