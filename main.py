from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch
from dataset import dataloaders
from VGG.VGG import VGG
from AlexNet.AlexNet import AlexNet

def define_params(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


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


def validate_VGG(num_epochs, device, classes, train_loader, val_loader, test_loader):
    print("Validating VGG model...")
    model = VGG(num_classes=len(classes))
    print(model)
    criterion, optimizer = define_params(model)

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, device, train_loader, optimizer, criterion)
        evaluate(model, device, val_loader, criterion)
    
    display_confusion_matrix(model, device, test_loader, classes)


def validate_AlexNet(num_epochs, device, classes, train_loader, val_loader, test_loader):
    print("Validating AlexNet model...")
    model = AlexNet(num_classes=len(classes))
    print(model)
    criterion, optimizer = define_params(model)

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, device, train_loader, optimizer, criterion)
        evaluate(model, device, val_loader, criterion)
    
    display_confusion_matrix(model, device, test_loader, classes)


if __name__ == "__main__":
    classes, train_loader, val_loader, test_loader = dataloaders(batch_size=32)
    print(f"Found {len(classes)} classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5

    validate_VGG(num_epochs, device, classes, train_loader, val_loader, test_loader)
    validate_AlexNet(num_epochs, device, classes, train_loader, val_loader, test_loader)