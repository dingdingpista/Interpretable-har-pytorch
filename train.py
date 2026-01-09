import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from models.simple_cnn import SimpleHARNet
from data.har_dataset import HARDataset

def train():
    # --- Settings ---
    batch_size = 32
    num_epochs = 5
    num_classes = 6

    # --- Device agnostic ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load real dataset ---
    train_dataset = HARDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Model, loss, optimizer ---
    model = SimpleHARNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Track loss for plotting ---
    epoch_losses = []

    # --- Training loop ---
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # --- Create results folder if missing ---
    os.makedirs("results", exist_ok=True)

    # --- Save model ---
    torch.save(model.state_dict(), "results/simple_cnn.pth")
    print("Model saved to results/simple_cnn.pth")

    # --- Plot training curve ---
    plt.figure()
    plt.plot(epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("results/training_curve.png")
    plt.close()

    print("Training curve saved to results/training_curve.png")

if __name__ == "__main__":
    train()
