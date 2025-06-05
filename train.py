
import os
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.optim as optim


from torchvision import transforms


from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

torch.cuda.empty_cache()
# preprocess data
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

#for training data add augmentations 

img_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=60),                     # Rotation
    transforms.RandomResizedCrop(224, scale=(0.7, 1.2)),       # Zoom (in/out)
    transforms.ColorJitter(brightness=0.4, contrast=0.3),      # Brightness and contrast
    transforms.RandomHorizontalFlip(),                         # Horizontal flip                           
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],           # EfficientNet normalization
                         std=[0.229, 0.224, 0.225]),
])
# Paths
train_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/train"
val_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/val_new"
test_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/test_new"

# Datasets
train_dataset = datasets.ImageFolder(root=train_path, transform=img_augmentation)
val_dataset = datasets.ImageFolder(root=val_path, transform=efficientnet_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=efficientnet_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=80, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False, num_workers=4)



# Load pretrained model without the final layer and add a layer as the classifier 
base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')


class EfficientNet_Vehicle(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.2):
        super(EfficientNet_Vehicle, self).__init__()
        self.base = base_model  # efficientnet_b4 backbone
        self.bn = nn.BatchNorm1d(base_model.num_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(base_model.num_features, num_classes)

    def forward(self, x):
        x = self.base(x)             
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Our model
num_classes = len(train_dataset.classes)  
model = EfficientNet_Vehicle(base_model, num_classes=num_classes)

# Move model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003)
# Separate parameters
classifier_params = list(model.fc.parameters()) + list(model.bn.parameters()) + list(model.dropout.parameters())
backbone_params = list(model.base.parameters())

# Create optimizer with different learning rates
optimizer = optim.Adam([
    {'params': backbone_params, 'lr': 0.0003},         # Lower LR for pretrained EfficientNet
    {'params': classifier_params, 'lr': 0.001}         # Higher LR for new classifier
])

# Training 
import time
epochs = 10
# Create a directory to save checkpoints
os.makedirs("checkpoints", exist_ok=True)

best_val_acc = 0.0  # Track the best validation accuracy
start = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)

            idx_to_class_val = val_dataset.classes
            idx_to_class_train = train_dataset.classes


            labels_names = [idx_to_class_val[idx] for idx in labels]
            predicted_names = [idx_to_class_train[idx] for idx in predicted]
            val_correct += sum(p == t for p, t in zip(predicted_names, labels_names))


    val_loss /= len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/model_epoch{epoch+1}_valacc{val_acc:.2f}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    # Savethe best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = "checkpoints/best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated and saved at epoch {epoch+1}")

print("Total time for training: ", time.time() - start)