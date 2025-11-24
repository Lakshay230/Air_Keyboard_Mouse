import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from tqdm import tqdm

# --- Configuration ---
NUM_CLASSES = 62
BATCH_SIZE = 512
EPOCHS = 20
MODEL_SAVE_NAME = "emnist_final_model.pth"

# --- 1. Define Helper Function (Global) ---
def get_mapping_dict():
    """
    Maps 62 classes down to 47 by merging ambiguous characters.
    (e.g., 'o' and 'O' become the same class).
    """
    mapping = {}
    # 0-9 (Numbers) -> Keep 0-9
    for i in range(10): mapping[i] = i
    # A-Z (10-35) -> Keep as is
    for i in range(10, 36): mapping[i] = i
    
    # a-z (36-61) -> Merge overlapping ones into Uppercase IDs
    merge_pairs = {
        38: 12, # c -> C
        44: 18, # i -> I
        45: 19, # j -> J
        46: 20, # k -> K
        47: 21, # l -> L
        48: 22, # m -> M
        50: 24, # o -> O
        51: 25, # p -> P
        54: 28, # s -> S
        56: 30, # u -> U
        57: 31, # v -> V
        58: 32, # w -> W
        59: 33, # x -> X
        60: 34, # y -> Y
        61: 35  # z -> Z
    }
    for i in range(36, 62):
        if i in merge_pairs:
            mapping[i] = merge_pairs[i]
        else:
            mapping[i] = i # Keep distinct
    return mapping

# --- 2. Define Model Class (Global) ---
class WiderCNN(nn.Module):
    def __init__(self):
        super(WiderCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2) 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Everything operational MUST be inside here for Windows
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomRotation(10) 
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Checking Dataset...")
    full_train_dataset = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform_train)
    test_dataset = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform_test)

    # Split
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model Setup ---
    model = WiderCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # ==========================================
    #        TRAINING / LOADING SWITCH
    # ==========================================
    TRAIN_MODE = False  # <--- Set FALSE to skip training and just evaluate
    # ==========================================

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0

    if TRAIN_MODE:
        print("Starting High-Accuracy Training...")
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
            
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)

            # Validation
            model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(avg_val_loss)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_NAME)
                print(f"✅ Saved Best Model")

        # Plot curves only if we trained
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(train_losses, label='Train', color='blue')
        ax[0].plot(val_losses, label='Val', color='orange')
        ax[0].set_title('Loss')
        ax[1].plot(train_accs, label='Train', color='green')
        ax[1].plot(val_accs, label='Val', color='red')
        ax[1].set_title('Accuracy')
        plt.show()

    else:
        print("Skipping training. Loading existing model...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_NAME))
            print("✅ Model loaded successfully.")
        except FileNotFoundError:
            print("❌ Error: Model file not found! Set TRAIN_MODE = True to train first.")
            exit()

    # --- 4. FINAL EVALUATION (Runs in both modes) ---
    
    # A. Standard Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    label_map = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_normalized, xticklabels=label_map, yticklabels=label_map, cmap='Blues', square=True, cbar=False)
    plt.title("Confusion Matrix")
    plt.show()

    # B. SMART EVALUATION (The new 95% logic)
    print("\nRunning Smart Evaluation (Merging ambiguous classes)...")
    mapping = get_mapping_dict()
    correct_strict = 0
    correct_merged = 0
    total_smart = 0
    
    # Re-run eval or reuse predictions (Running fresh loop for clarity)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Smart Eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total_smart += labels.size(0)
            correct_strict += (predicted == labels).sum().item()
            
            pred_np = predicted.cpu().numpy()
            label_np = labels.cpu().numpy()
            
            for p, l in zip(pred_np, label_np):
                if mapping.get(p, p) == mapping.get(l, l):
                    correct_merged += 1
    
    print("--------------------------------------------------")
    print(f"Final Strict Accuracy (62 Classes): {100 * correct_strict / total_smart:.2f}%")
    print(f"Final Merged Accuracy (47 Classes): {100 * correct_merged / total_smart:.2f}%")
    print("--------------------------------------------------")