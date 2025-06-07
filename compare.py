import os
import torch
import torch.nn.functional as F
# from train.py import EfficientNet_Vehicle, base_model,num_classes
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import time
import timm

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
# preprocess data
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

# Paths

test_path = "/home/kalinga/Venuri_Vision/efficientnet/data/test_new"

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os

class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            # Skip unreadable image by returning None (caller must handle)
            return None

        if self.transform:
            image = self.transform(image)

        return image, img_path  # or return just `image` if path not needed

# Datasets

test_dataset = TestImageDataset(test_path, transform=efficientnet_transform)
# Read class names from file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Number of classes
num_classes = len(class_names)
# DataLoaders

data_loader = DataLoader(test_dataset, batch_size=80, shuffle=False, num_workers=4)

# Load pretrained model without the final layer and add a layer as the classifier 
base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
model = EfficientNet_Vehicle(base_model, num_classes=num_classes)

# Move model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


print("Original Model Size:", os.path.getsize("best_model.pth") / 1e6, "MB")
print("ONNX Model Size:", os.path.getsize("model.onnx") / 1e6, "MB")
print("TensorRT Engine Size:", os.path.getsize("model_int8.engine") / 1e6, "MB")

def benchmark_pytorch(model, data_loader, device, reps=100):
    model.eval()
    model.to(device)
    start = time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= reps:
                break
            _ = model(x.to(device))
    end = time.time()
    fps = reps / (end - start)
    print(f"[PyTorch] FPS: {fps:.2f}")
    return fps

import onnxruntime as ort

def benchmark_onnx(onnx_path, data_loader, reps=100):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    start = time.time()
    for i, (x, _) in enumerate(data_loader):
        if i >= reps:
            break
        ort_inputs = {input_name: x.numpy()}
        _ = session.run(None, ort_inputs)
    end = time.time()
    fps = reps / (end - start)
    print(f"[ONNX] FPS: {fps:.2f}")
    return fps


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np



def load_engine(trt_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_trt(engine, data_loader, reps=100):
    context = engine.create_execution_context()
    input_binding_name = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
    output_binding_name = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]

    input_shape = engine.get_binding_shape(input_binding_idx)
    output_shape = engine.get_binding_shape(output_binding_idx)

    input_size = np.prod(input_shape).item() * np.dtype(np.float32).itemsize
    output_size = np.prod(output_shape).item() * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    start = time.time()
    for i, (x, _) in enumerate(data_loader):
        if i >= reps:
            break
        np_input = x.numpy().astype(np.float32)
        cuda.memcpy_htod(d_input, np_input)
        context.execute_v2([int(d_input), int(d_output)])
    end = time.time()
    fps = reps / (end - start)
    print(f"[TensorRT INT8] FPS: {fps:.2f}")
    return fps


fps_pytorch = benchmark_pytorch(model, data_loader, device)
fps_onnx = benchmark_onnx("model.onnx", data_loader)



import tensorrt as trt

TRT_LOGGER = trt.Logger()

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
engine = load_engine("model_int8.engine")
engine = load_engine("model_int8.engine")

# for i in range(engine.num_bindings):
#     name = engine.get_binding_name(i)
#     is_input = engine.binding_is_input(i)
#     print(f"{'Input' if is_input else 'Output'} {i}: {name}")


fps_trt = benchmark_trt(engine, data_loader)

# Visualization
import matplotlib.pyplot as plt

labels = ["PyTorch", "ONNX", "TensorRT INT8"]
fps_values = [fps_pytorch, fps_onnx, fps_trt]

plt.bar(labels, fps_values, color=["blue", "orange", "green"])
plt.ylabel("FPS")
plt.title("Inference Speed Comparison")
plt.show()
