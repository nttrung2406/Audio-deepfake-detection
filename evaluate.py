import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from data_reader import TestDataset, test_path
from model import CNNNetwork

test_dataset = TestDataset(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
results = []

model = CNNNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for audio, audio_file in test_dataloader:
        audio = audio.to(device)
        outputs = model(audio)

        _, preds = torch.max(outputs, 1)

        for i in range(len(outputs)):
            results.append((audio_file[i], torch.exp(outputs[i][1]).item()))