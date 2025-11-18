import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- Configuration ---
NUM_CLASSES = 62  # EMNIST ByClass has 62 classes (0-9, A-Z, a-z)
MODEL_SAVE_NAME = "emnist_byclass_cnn.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data loading (EMNIST ByClass) ---
# We just load the raw data. The EMNIST split is already rotated/flipped.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading EMNIST 'byclass' dataset...")
train_dataset = datasets.EMNIST(
    root='./data', 
    split='byclass',
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = datasets.EMNIST(
    root='./data', 
    split='byclass',
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Dataset loaded.")

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        # --- FIX ---
        # We do NO transforms here.
        # The model will learn from the raw EMNIST data as-is.
        # (The original data is already rotated/flipped)
        # --- END FIX ---
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model setup
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
print("Starting training...")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# --- Evaluation ---
print("Starting evaluation...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# --- Save the model ---
torch.save(model.state_dict(), MODEL_SAVE_NAME)
print(f"✅ Model saved as {MODEL_SAVE_NAME}")