import os
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight

torch.cuda.empty_cache()

# -------------------------
# 1. Data preparation
# -------------------------

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=60),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.2)),
    transforms.ColorJitter(brightness=0.4, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/train"
val_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/val_new"
test_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/test_new"

# Load full datasets with ImageFolder (labels are full class names like "Toyota_Camry_2018")
train_dataset_full = datasets.ImageFolder(root=train_path)
val_dataset_full = datasets.ImageFolder(root=val_path)
test_dataset_full = datasets.ImageFolder(root=test_path)

# Extract class names (full classes)
full_classes = train_dataset_full.classes  # list of full class names

# Build a unique sorted list of make+model classes by removing the year from full classes
def extract_make_model(full_name):
    # Example: "Toyota_Camry_2018" -> "Toyota_Camry"
    # Assumes last underscore separates year
    parts = full_name.rsplit('_', 1)
    return parts[0]

make_model_classes = sorted(list(set(extract_make_model(c) for c in full_classes)))

# Build mappings from class name to index for full and make+model classes
full_class_to_idx = {c: i for i, c in enumerate(full_classes)}
make_model_to_idx = {c: i for i, c in enumerate(make_model_classes)}

# Create the mapping matrix M (make_model_count x full_class_count)
N_full = len(full_classes)
N_mm = len(make_model_classes)

M = torch.zeros((N_mm, N_full), dtype=torch.float32)
for j, full_cls in enumerate(full_classes):
    mm_cls = extract_make_model(full_cls)
    i = make_model_to_idx[mm_cls]
    M[i, j] = 1.0

# -------------------------
# 2. Custom Dataset to include make_model labels
# -------------------------

class VehicleDatasetWithMakeModel(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, full_label = self.base_dataset[idx]
        full_label_name = full_classes[full_label]
        make_model_label_name = extract_make_model(full_label_name)

        full_label_idx = full_label
        make_model_label_idx = make_model_to_idx[make_model_label_name]

        if self.transform:
            image = self.transform(image)

        return image, full_label_idx, make_model_label_idx

# Prepare datasets with transforms
train_dataset = VehicleDatasetWithMakeModel(train_dataset_full, transform=img_augmentation)
val_dataset = VehicleDatasetWithMakeModel(val_dataset_full, transform=efficientnet_transform)
test_dataset = VehicleDatasetWithMakeModel(test_dataset_full, transform=efficientnet_transform)

train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=80, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False, num_workers=4)

# -------------------------
# 3. Model
# -------------------------

base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')

class EfficientNet_Vehicle(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.2):
        super(EfficientNet_Vehicle, self).__init__()
        self.base = base_model
        self.bn = nn.BatchNorm1d(base_model.num_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(base_model.num_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

num_classes = N_full  # number of full classes
model = EfficientNet_Vehicle(base_model, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
M = M.to(device)

# -------------------------
# 4. Loss, optimizer
# -------------------------

# Class weights for full classes
weights = compute_class_weight('balanced', classes=np.arange(N_full),
                               y=[label for _, label, _ in train_dataset])
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion_full = nn.CrossEntropyLoss(weight=class_weights)

# For make_model loss, we don't use weights here but you can add if needed

optimizer = optim.Adam(model.parameters(), lr=3e-4)

# -------------------------
# 5. Training loop
# -------------------------

epochs = 10
alpha = 0.5  # weight for make_model loss

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_full = 0
    total = 0

    for inputs, full_labels, make_model_labels in train_loader:
        inputs = inputs.to(device)
        full_labels = full_labels.to(device)
        make_model_labels = make_model_labels.to(device)

        optimizer.zero_grad()
        outputs_full = model(inputs)  # logits for full classes

        # Full loss
        loss_full = criterion_full(outputs_full, full_labels)

        # Compute make_model probabilities by aggregating full class probs
        probs_full = torch.softmax(outputs_full, dim=1)
        probs_make_model = torch.matmul(probs_full, M.t())  # shape: [batch, N_mm]

        # Make model loss: NLLLoss on log probs
        log_probs_make_model = torch.log(probs_make_model + 1e-8)
        loss_make_model = F.nll_loss(log_probs_make_model, make_model_labels)

        loss = alpha * loss_full + alpha * loss_make_model

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted_full = torch.max(outputs_full, 1)
        correct_full += (predicted_full == full_labels).sum().item()
        total += full_labels.size(0)

    train_loss = running_loss / total
    train_acc = 100.0 * correct_full / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0



    with torch.no_grad():
        for inputs, full_labels, make_model_labels in val_loader:
            inputs = inputs.to(device)
            full_labels = full_labels.to(device)
            # make_model_labels = make_model_labels.to(device)

            outputs_full = model(inputs)

            loss_full = criterion_full(outputs_full, full_labels)
            # probs_full = torch.softmax(outputs_full, dim=1)
            # probs_make_model = torch.matmul(probs_full, M.t())
            # log_probs_make_model = torch.log(probs_make_model + 1e-8)
            # loss_make_model = F.nll_loss(log_probs_make_model, make_model_labels)

            loss = loss_full 

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max( outputs_full, 1)
            val_total += full_labels.size(0)

            idx_to_class_val = val_dataset_full.classes
            idx_to_class_train = train_dataset_full.classes


            labels_names = [idx_to_class_val[idx] for idx in full_labels]
            predicted_names = [idx_to_class_train[idx] for idx in predicted]
            val_correct += sum(p == t for p, t in zip(predicted_names, labels_names))


    val_loss /= len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save checkpoint
    checkpoint_path = f"checkpoints_2_label/model_epoch{epoch+1}_valacc{val_acc:.2f}.pth"
    os.makedirs("checkpoints_2_label", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

print("Training complete.")
