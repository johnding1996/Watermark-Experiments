from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import torch


def evaluate_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_classifier(
    train_loader, test_loader, device, learning_rate, num_epochs, verbose=False
):
    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device)
    # Change last layer to output 2 classes (for your merged dataset)
    model.fc = nn.Linear(model.fc.in_features, 2).to(device)

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        # Evaluate on the test set
        model.eval()
        test_acc = evaluate_accuracy(model, test_loader, device)

        if verbose:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.3e} Test ACC: {test_acc:.2f}%"
            )
    return model


def load_classifier(save_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Change last layer to output 2 classes (for your merged dataset)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(save_path))
    return model
