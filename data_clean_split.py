import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import random
import shutil

dataset_root = Path('/home/kalinga/DomainGeneralization/venuri/Vehicle_Make_Model_Recognition/VMMRdb')

#data set loader to load only the classes with valid data 


class Clean_Dataset(Dataset):
    def __init__(self, root, transform=None, min_images=5):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for class_dir in sorted(self.root.iterdir()):
            if class_dir.is_dir():
                images = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_exts]
                if len(images) >= min_images:
                    class_name = class_dir.name
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)
                    class_idx = self.class_to_idx[class_name]
                    for img_path in images:
                        self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

dataset = Clean_Dataset(dataset_root)

labels = np.array([label for _, label in dataset.samples])
unique_classes, class_counts = np.unique(labels, return_counts=True)

# 1) Separate single-sample classes and multi-sample classes
single_sample_classes = unique_classes[class_counts == 1]
multi_sample_classes = unique_classes[class_counts > 1]

# Indices for single-sample classes => all go to train
single_sample_indices = [i for i, lbl in enumerate(labels) if lbl in single_sample_classes]

# Indices and labels for multi-sample classes
multi_sample_indices = [i for i, lbl in enumerate(labels) if lbl in multi_sample_classes]
multi_sample_labels = labels[multi_sample_indices]

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 2) Stratified split multi-sample classes into train and temp(val+test)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=42)
train_multi_idx, temp_idx = next(sss1.split(np.zeros(len(multi_sample_labels)), multi_sample_labels))

# Map back indices to original dataset
train_multi_indices = np.array(multi_sample_indices)[train_multi_idx]
temp_indices = np.array(multi_sample_indices)[temp_idx]
temp_labels = labels[temp_indices]

# 3) Split temp into val and test stratified as much as possible
# But handle classes with only 1 sample in temp by random assignment
val_indices = []
test_indices = []

# Find classes in temp and their indices
temp_classes, temp_class_counts = np.unique(temp_labels, return_counts=True)

for cls in temp_classes:
    cls_indices = temp_indices[temp_labels == cls]
    if len(cls_indices) == 1:
        # Randomly assign this single sample to val or test
        if random.random() < 0.5:
            val_indices.append(cls_indices[0])
        else:
            test_indices.append(cls_indices[0])
    else:
        # For classes with multiple samples, do stratified split
        cls_labels = labels[cls_indices]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
        val_cls_idx, test_cls_idx = next(sss2.split(np.zeros(len(cls_labels)), cls_labels))
        val_indices.extend(cls_indices[val_cls_idx])
        test_indices.extend(cls_indices[test_cls_idx])

# Combine train indices
train_indices = np.concatenate([single_sample_indices, train_multi_indices])

# Create dataset subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

#save the splits to folders 

def get_original_samples(subset):
    return [subset.dataset.samples[i] for i in subset.indices]

# Save images to folders
def save_to_folder(samples, root_dir):
    for img_path, label in samples:
        class_name = dataset.classes[label]
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

# Create splits 
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val_new", exist_ok=True)
os.makedirs("data/test_new", exist_ok=True)

# Save datasets
save_to_folder(get_original_samples(train_dataset), "data/train")
save_to_folder(get_original_samples(val_dataset), "data/val_new")
save_to_folder(get_original_samples(test_dataset), "data/test_new")

print("Images successfully saved to data/train, data/val, data/test.")