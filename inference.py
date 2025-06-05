import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from train.py import EfficientNet_Vehicle, base_model,num_classes
import os
import torch
import timm
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


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
test_path = "/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/data/test_new"

# Datasets
train_dataset = datasets.ImageFolder(root=train_path, transform=img_augmentation)
test_dataset = datasets.ImageFolder(root=test_path, transform=efficientnet_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False, num_workers=4)
model = EfficientNet_Vehicle(base_model, num_classes=num_classes)

# Move model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


model.load_state_dict(torch.load("/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/checkpoints/best_model.pth"))


model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  
        probs = F.softmax(outputs, dim=1)
        
        all_preds.append(probs.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Convert to numpy for sklearn
y_true = all_labels.numpy()
y_pred_top1 = all_preds.argmax(dim=1).numpy()

# Assuming test_dataset.classes is a list: index -> class name
idx_to_class_test = test_dataset.classes
idx_to_class_train = train_dataset.classes

# Convert all true and predicted indices to class names
y_true_names = [idx_to_class_test[idx] for idx in y_true]
y_pred_names = [idx_to_class_train[idx] for idx in y_pred_top1]

# Calculate accuracy by comparing names
correct = sum(p == t for p, t in zip(y_pred_names, y_true_names))
accuracy = correct / len(y_true_names) * 100
print(f"Top-1 Accuracy (by class names): {accuracy:.2f}%")

# For top-5 accuracy
top5_preds = all_preds.topk(5, dim=1).indices.numpy()
top5_preds_names = [[idx_to_class_train[idx] for idx in top5] for top5 in top5_preds]

top5_correct = 0
for true_name, pred_names in zip(y_true_names, top5_preds_names):
    if true_name in pred_names:
        top5_correct += 1
top5_acc = top5_correct / len(y_true_names) * 100
print(f"Top-5 Accuracy (by class names): {top5_acc:.2f}%")




f1_macro = f1_score(y_true_names, y_pred_names, average='macro')
f1_weighted = f1_score(y_true_names, y_pred_names, average='weighted')
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")


# Suppose class_names = train_dataset.classes
class_names = train_dataset.classes
# Encode class names if needed

le = LabelEncoder()
le.fit(class_names)

# Convert string labels to integer indices
y_true_int = le.transform(y_true_names)
y_pred_int = le.transform(y_pred_names)

# Compute confusion matrix
cm = confusion_matrix(y_true_int, y_pred_int)

# Plot the matrix
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# Show labels (optionally limit for clarity)
tick_marks = np.arange(len(le.classes_))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=6)
ax.set_yticklabels(le.classes_, fontsize=6)

ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Confusion Matrix')


threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j] > 0:  # Only annotate non-zero cells
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=4)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("Confusion matrix saved as 'confusion_matrix_inference.png'")