# TODO: Train watermark detector
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from tqdm.notebook import tqdm, trange


# Train detector
def train_detector(dataloader, device, train_paras):
    learning_rate = train_paras["learning_rate"]
    num_epochs = train_paras["num_epochs"]
    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.to(device)
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train loop
    for epoch in tqdm(range(num_epochs), desc="Train Classifier"):
        model.train()
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}")
    return model
