import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from data_reader import AudioDataset, train_path
from model import CNNNetwork
import os

dataset = AudioDataset(train_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNNNetwork()
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Directory to save the model weights
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for audio, labels in dataloader:
        audio = audio.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += ((outputs > 0.5).float() == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

print("Training complete!")
