# evaluation.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import your dataset and model
from data.har_dataset import HARDataset
from models.simple_cnn import SimpleHARNet

# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_classes = 6
activity_names = ["Walking", "Walking Upstairs", "Walking Downstairs",
                  "Sitting", "Standing", "Laying"]

# --- Load test dataset ---
test_dataset = HARDataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Load trained model ---
model = SimpleHARNet(num_classes=num_classes).to(device)
# If you saved a checkpoint:
checkpoint_path = "/content/har_interpretable_dl/results/simple_cnn.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()

all_preds = []
all_labels = []

# --- Evaluation loop ---
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Overall Metrics ---
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro'
)

print(f"\n--- Overall Metrics ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# --- Per-Class Metrics ---
print("\n--- Per-Class Metrics ---")
precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

for i, name in enumerate(activity_names):
    print(f"{name:20s} | Precision={precision_per[i]:.3f}, Recall={recall_per[i]:.3f}, F1={f1_per[i]:.3f}")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=activity_names, yticklabels=activity_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
# Save figure for README
if not os.path.exists("results"):
    os.makedirs("results")
plt.savefig("results/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved to results/confusion_matrix.png")
