import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# --- Configuration ---
NUM_CLASSES = 62
BATCH_SIZE = 256  # ResNet is heavy, 256 is safer than 512
EPOCHS = 25       # Deep networks need time to converge
MODEL_SAVE_NAME = "emnist_resnet18_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading ---
# We use slightly heavier augmentation for ResNet to prevent overfitting
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)) # Shift, Zoom, Rotate
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading EMNIST Data...")
full_train_dataset = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform_train)
test_dataset = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform_test)

# Split
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- 2. The "Hacked" ResNet Architecture ---
def get_custom_resnet():
    # Load standard ResNet18 structure (not pre-trained weights, we train from scratch)
    model = models.resnet18(weights=None)
    
    # MODIFICATION 1: The Input Layer
    # Standard ResNet: Conv2d(3, 64, 7x7, stride 2) -> Expects 224x224 RGB
    # Our Hack:        Conv2d(1, 64, 3x3, stride 1) -> Works on 28x28 Grayscale
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # MODIFICATION 2: Remove the first MaxPool
    # Pooling reduces size. 28x28 is already small. If we pool too early, we lose detail.
    model.maxpool = nn.Identity()
    
    # MODIFICATION 3: The Output Layer
    # Change from 1000 ImageNet classes to 62 EMNIST classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    return model

# --- 3. Helper: 47-Class Mapping ---
def get_mapping_dict():
    mapping = {}
    for i in range(10): mapping[i] = i
    for i in range(10, 36): mapping[i] = i
    # Merge Lowercase into Uppercase
    merge_pairs = {38:12, 44:18, 45:19, 46:20, 47:21, 48:22, 50:24, 51:25, 54:28, 56:30, 57:31, 58:32, 59:33, 60:34, 61:35}
    for i in range(36, 62):
        if i in merge_pairs: mapping[i] = merge_pairs[i]
        else: mapping[i] = i
    return mapping

# --- 4. Execution Block ---
if __name__ == '__main__':
    model = get_custom_resnet().to(device)
    
    # ResNet often likes SGD with Momentum better than Adam
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine Scheduler (Smoother than the Plateau one)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("Starting ResNet Training...")
    
    train_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Scheduler Step
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Loop (Check Merged Accuracy during training!)
        model.eval()
        correct_merged = 0
        total = 0
        mapping = get_mapping_dict()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Fast Numpy conversion for mapping check
                pred_np = predicted.cpu().numpy()
                label_np = labels.cpu().numpy()
                total += labels.size(0)
                
                for p, l in zip(pred_np, label_np):
                    if mapping.get(p, p) == mapping.get(l, l):
                        correct_merged += 1
                        
        val_acc = 100 * correct_merged / total
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Merged Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            print("✅ Saved Best ResNet Model")

    print(f"Training Complete. Best Merged Accuracy: {best_val_acc:.2f}%")

    # --- 5. VISUALIZATION ---
    
    # Plot 1: Loss vs Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_accs, label="Merged Val Accuracy")
    plt.legend()
    plt.title(f"ResNet Training (Best Acc: {best_val_acc:.2f}%)")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.show()

    # Plot 2: Confusion Matrix
    print("Generating Confusion Matrix...")
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_NAME))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Eval Matrix"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # You need seaborn for this part
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np

    label_map = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize to see percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_normalized, xticklabels=label_map, yticklabels=label_map, cmap='Blues', square=True, cbar=False)
    plt.title("ResNet Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()