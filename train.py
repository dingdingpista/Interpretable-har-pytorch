import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.simple_cnn import SimpleHARNet
from data.har_dataset import HARDataset

def train():
    X_train = torch.randn(100, 9, 128)
    y_train = torch.randint(0, 6, (100,))

    dataset = HARDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleHARNet(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
