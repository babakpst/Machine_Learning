# PyTorch Deep Learning — Comprehensive Course Notes

> Synthesized from:
> - Course 14: PyTorch for Deep Learning (C1–C4)
> - Course 15: PyTorch Techniques and Ecosystem Tools

---

## Table of Contents

1. [Tensors](#1-tensors)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Loop](#4-training-loop)
5. [Loss Functions & Optimizers](#5-loss-functions--optimizers)
6. [Learning Rate Scheduling](#6-learning-rate-scheduling)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Convolutional Neural Networks (CNNs)](#8-convolutional-neural-networks-cnns)
9. [Transfer Learning & Fine-Tuning](#9-transfer-learning--fine-tuning)
10. [Regularization Techniques](#10-regularization-techniques)
11. [Hyperparameter Tuning](#11-hyperparameter-tuning)
12. [DataLoader Optimization](#12-dataloader-optimization)
13. [TorchVision Utilities](#13-torchvision-utilities)
14. [Advanced Techniques](#14-advanced-techniques)
15. [Debugging & Best Practices](#15-debugging--best-practices)
16. [Quick Reference Cheatsheet](#16-quick-reference-cheatsheet)

---

## 1. Tensors

Tensors are the fundamental data structure in PyTorch — n-dimensional arrays with GPU support and autograd.

### Creation

```python
import torch

# From Python data
t = torch.tensor([1.0, 2.0, 3.0])
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# From NumPy
import numpy as np
t = torch.from_numpy(np.array([1, 2, 3]))

# Predefined shapes
torch.zeros(3, 4)
torch.ones(3, 4)
torch.rand(3, 4)          # Uniform [0, 1)
torch.randn(3, 4)         # Normal distribution
torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
torch.eye(3)              # Identity matrix
```

### Reshaping

```python
t.shape                        # View dimensions
t.reshape(2, 3)                # Reshape (returns view when possible)
t.view(2, -1)                  # View (must be contiguous)
t.unsqueeze(0)                 # Add dimension at index 0
t.squeeze()                    # Remove all size-1 dimensions
t.squeeze(0)                   # Remove size-1 dim at index 0
t.transpose(0, 1)              # Swap two dimensions
t.permute(2, 0, 1)             # Reorder all dimensions
t.flatten()                    # Collapse to 1D
t.contiguous()                 # Make memory contiguous
```

**Tip:** `unsqueeze(0)` is commonly used to add a batch dimension — convert `[C, H, W]` → `[1, C, H, W]` for inference.

### Indexing & Slicing

```python
t[1]                           # Row 1
t[:, 2]                        # Column 2 (all rows)
t[0:2, 1:]                     # Slice rows and cols
t[t > 5]                       # Boolean mask
t[[0, 2], :]                   # Fancy indexing
t[0, 1].item()                 # Extract Python scalar
```

### Operations

```python
# Arithmetic (element-wise)
a + b, a - b, a * b, a / b

# Matrix multiplication
torch.matmul(a, b)     # or a @ b

# Statistics
t.mean(), t.std(), t.sum(), t.max(), t.min()
t.mean(dim=0)          # Along specific dimension

# Type casting
t.float(), t.int(), t.long(), t.bool()

# Comparison & logical
a == b, a > b, a < b
(mask1) & (mask2)      # AND
(mask1) | (mask2)      # OR

# Concatenation / stacking
torch.cat([a, b], dim=1)   # Concatenate
torch.stack([a, b], dim=0) # Stack (new dimension)
```

### Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t = t.to(device)
model = model.to(device)

# Always move both data AND model to same device
inputs, labels = inputs.to(device), labels.to(device)
```

**Tip:** Check device with `t.device`. Mixing CPU/GPU tensors causes a runtime error.

---

## 2. Data Pipeline

The standard PyTorch data pipeline: **Dataset → DataLoader → Transforms**.

### Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()   # List of (path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self._load_item(idx)
        if self.transform:
            image = self.transform(image)
        return image, label
```

**Key rules:**
- `__len__` must return the total number of samples.
- `__getitem__` must return a `(sample, label)` tuple.
- Labels from MATLAB/R files are often 1-indexed — subtract 1 for Python.

### Robust Dataset (Error Handling)

```python
def __getitem__(self, idx):
    for _ in range(len(self)):
        try:
            image = Image.open(self.paths[idx]).convert('RGB')
            image.verify()          # Check file integrity
            image = Image.open(self.paths[idx])  # Re-open after verify
            if min(image.size) < 32:
                raise ValueError('Image too small')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f'Skipping {idx}: {e}')
            idx = (idx + 1) % len(self)
```

### DataLoader

```python
from torch.utils.data import DataLoader, random_split

# Split dataset
train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
```

**`shuffle=True`** only on training. Never shuffle validation/test — it doesn't change results but is misleading.

### Applying Different Transforms per Split

```python
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label

train_dataset = SubsetWithTransform(train_set, train_transform)
val_dataset   = SubsetWithTransform(val_set,   val_transform)
```

### Transforms

```python
from torchvision import transforms

# Training transform (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),                          # Scales [0,255] → [0,1], HWC → CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # ImageNet statistics
])

# Validation/test transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Order matters:**
1. Geometric transforms (Resize, Crop, Flip, Rotate)
2. Color augmentations (ColorJitter)
3. `ToTensor()` — converts PIL to tensor and scales to [0, 1]
4. `Normalize()` — zero-center and scale by std

### Computing Dataset Mean & Std

```python
# 2-pass algorithm
def compute_mean_std(loader):
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    total = 0
    for images, _ in loader:
        batch = images.size(0)
        images = images.view(batch, images.size(1), -1)  # [B, C, H*W]
        mean += images.mean(2).sum(0)
        std  += images.std(2).sum(0)
        total += batch
    return mean / total, std / total
```

**Tip:** For grayscale→RGB conversion for pretrained models: `transforms.Grayscale(num_output_channels=3)`.

---

## 3. Model Architecture

### `nn.Sequential` (Simple)

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
```

### `nn.Module` (Recommended)

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        return self.fc2(x)
```

### Common Layers

| Layer | Usage | Notes |
|---|---|---|
| `nn.Linear(in, out)` | Fully connected | Basic building block |
| `nn.Conv2d(in_ch, out_ch, kernel, padding)` | 2D convolution | Use `padding=1` with `kernel=3` to preserve size |
| `nn.MaxPool2d(k, stride)` | Downsampling | `k=2, stride=2` halves H×W |
| `nn.AvgPool2d(k)` | Averaging pool | Smoother than MaxPool |
| `nn.BatchNorm1d/2d(features)` | Batch normalization | Place after Linear/Conv, before activation |
| `nn.Dropout(p)` | Regularization | Deactivated in `model.eval()` |
| `nn.Flatten()` | Reshape | `[B, C, H, W]` → `[B, C*H*W]` |
| `nn.ReLU()` | Activation | Most common; `inplace=True` saves memory |
| `nn.GELU()` | Activation | Used in transformers |
| `nn.Embedding(vocab, dim)` | Token embedding | For NLP tasks |

### CNN Building Pattern

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels, 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 → 64 channels, 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 → 128 channels, 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**Tip — Calculate classifier input size:** Pass a dummy tensor through the feature extractor:
```python
dummy = torch.randn(1, 3, 32, 32)
flat_size = model.features(dummy).shape[1]   # After flatten
```

---

## 4. Training Loop

### Standard Pattern

```python
import torch
import torch.nn as nn
from torch import optim

# Setup
model = MyModel(...).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

train_losses, val_losses, val_accs = [], [], []

for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc  = correct / len(val_loader.dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    # Step scheduler AFTER evaluation
    scheduler.step(epoch_val_acc)

    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} | '
          f'Val Acc: {epoch_val_acc:.4f} | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
```

### Key Training Steps (order matters)

1. `optimizer.zero_grad()` — clear previous gradients
2. `outputs = model(inputs)` — forward pass
3. `loss = criterion(outputs, labels)` — compute loss
4. `loss.backward()` — compute gradients
5. `optimizer.step()` — update weights

**Never forget `zero_grad()`!** PyTorch accumulates gradients by default.

### Saving & Loading Checkpoints

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': best_val_loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 5. Loss Functions & Optimizers

### Loss Functions

| Task | Loss | Notes |
|---|---|---|
| Regression | `nn.MSELoss()` | Mean squared error |
| Regression | `nn.L1Loss()` | Mean absolute error; more robust to outliers |
| Binary classification | `nn.BCEWithLogitsLoss()` | Combines sigmoid + BCE (numerically stable) |
| Multi-class classification | `nn.CrossEntropyLoss()` | Combines log-softmax + NLL; **do not apply softmax before** |

```python
# CrossEntropyLoss expects raw logits, not softmax output
outputs = model(inputs)               # Shape: [B, num_classes]
loss = criterion(outputs, labels)     # labels: [B] with class indices (not one-hot)
```

### Optimizers

```python
# SGD — simple, good for fine-tuning, momentum helps escape local minima
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — adaptive LR, converges faster, good default
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW — Adam with decoupled weight decay (better for transformers)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

**Tip:** Lower `lr` (1e-5 to 1e-4) when fine-tuning pretrained models to preserve learned features.

### Weight Decay (L2 Regularization)

Adding `weight_decay` to the optimizer applies L2 regularization — penalizes large weights and reduces overfitting. Typical value: `1e-4`.

---

## 6. Learning Rate Scheduling

Always call `scheduler.step()` after the optimizer step (and after evaluation for plateau-based schedulers).

### StepLR

```python
# Reduce LR by gamma every step_size epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
# In loop (after optimizer.step):
scheduler.step()
```

### CosineAnnealingLR

```python
# Smooth cosine decay from initial LR to eta_min over T_max epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scheduler.step()
```

### ReduceLROnPlateau (Recommended)

```python
# Reduce when a metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
# In loop (after validation):
scheduler.step(val_accuracy)
```

### Get Current LR

```python
current_lr = optimizer.param_groups[0]['lr']
```

---

## 7. Evaluation & Metrics

### Accuracy (Manual)

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
```

### TorchMetrics (Recommended)

```python
import torchmetrics

accuracy  = torchmetrics.Accuracy(task='multiclass', num_classes=N, average='micro').to(device)
precision = torchmetrics.Precision(task='multiclass', num_classes=N, average='macro').to(device)
recall    = torchmetrics.Recall(task='multiclass', num_classes=N, average='macro').to(device)
f1        = torchmetrics.F1Score(task='multiclass', num_classes=N, average='macro').to(device)

# In evaluation loop
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        preds = outputs.argmax(dim=1)
        accuracy.update(preds, labels.to(device))
        f1.update(preds, labels.to(device))

print(f'Accuracy: {accuracy.compute():.4f}')
print(f'F1: {f1.compute():.4f}')
accuracy.reset()   # Reset for next epoch
```

### Averaging Modes

| Mode | Description | Use When |
|---|---|---|
| `micro` | Global across all samples (= standard accuracy) | Balanced datasets |
| `macro` | Average per class (equal weight) | Imbalanced datasets |
| `weighted` | Average weighted by class frequency | Imbalanced, care about larger classes |

### Plotting Loss & Accuracy

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss'); ax1.legend()

ax2.plot(val_accs, label='Val Accuracy', color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy'); ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Convolutional Neural Networks (CNNs)

### Core Building Blocks

| Block | Purpose | Example |
|---|---|---|
| `Conv2d` | Feature extraction | Detects edges, textures, shapes |
| `BatchNorm2d` | Training stability | Normalizes activations per batch |
| `ReLU` | Non-linearity | `max(0, x)` |
| `MaxPool2d` | Spatial downsampling | Reduces H×W by factor of 2 |
| `Dropout` | Regularization | Randomly zeros activations |
| `Flatten` | Bridge conv→linear | `[B, C, H, W]` → `[B, C*H*W]` |

### Spatial Dimension Formula

```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```

With `kernel=3, padding=1, stride=1`: output = input (preserved).
With `MaxPool2d(2, 2)`: output = input / 2.

### Overfitting Diagnosis

Signs of overfitting:
- Training loss keeps decreasing
- Validation loss stops decreasing or increases
- Large gap between train and val accuracy

**Fixes:**
- Add `Dropout` (p=0.3–0.5)
- Add `BatchNorm`
- Add data augmentation
- Reduce model capacity
- Add `weight_decay` to optimizer
- Use early stopping

---

## 9. Transfer Learning & Fine-Tuning

### Load a Pretrained Model

```python
import torchvision.models as models

# Modern API (recommended)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
```

### Replace the Classifier Head

```python
# ResNet-style (single fc layer)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# MobileNet/EfficientNet-style (Sequential classifier)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)
```

### Strategy 1 — Feature Extraction (Fastest)

```python
# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the new head
for param in model.fc.parameters():
    param.requires_grad = True

# Optimizer only gets trainable parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

Best for: small datasets, quick results.

### Strategy 2 — Fine-Tuning (Balanced, Two-Stage)

```python
# Stage 1: Train head only (same as Strategy 1, ~5 epochs)
# Stage 2: Unfreeze top layers and continue

# Unfreeze last layer group of the backbone
for param in model.layer4.parameters():    # ResNet
    param.requires_grad = True

# Use lower LR for fine-tuning
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(),     'lr': 1e-4},
])
```

Best for: medium datasets, best accuracy/cost ratio.

### Strategy 3 — Full Retraining

```python
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

Best for: large domain-shift, large dataset available.

### Standard Pretrained Input Preprocessing

```python
# ImageNet statistics — use for ALL torchvision pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

pretrained_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

**Tip:** Grayscale to 3-channel: `transforms.Grayscale(num_output_channels=3)` before ToTensor.

---

## 10. Regularization Techniques

| Technique | How | Effect |
|---|---|---|
| **Dropout** | `nn.Dropout(p=0.5)` | Prevents co-adaptation of neurons |
| **Weight Decay** | `weight_decay=1e-4` in optimizer | L2 penalty on weights |
| **Batch Normalization** | `nn.BatchNorm2d(C)` | Reduces internal covariate shift; acts as regularizer |
| **Data Augmentation** | Flips, rotations, ColorJitter | Increases effective dataset size |
| **Early Stopping** | Monitor val loss, stop if no improvement | Prevents overfitting on training set |
| **Label Smoothing** | `CrossEntropyLoss(label_smoothing=0.1)` | Softens confidence, improves calibration |

---

## 11. Hyperparameter Tuning

### Key Hyperparameters and Search Ranges

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| Learning rate | `1e-5` – `1e-1` | Most impactful; often best at `1e-3` |
| Batch size | 16, 32, 64, 128 | Larger = more stable gradients; may need LR adjustment |
| Dropout rate | 0.1 – 0.5 | Tune based on overfitting severity |
| Hidden units | 64, 128, 256, 512 | Power of 2 for GPU efficiency |
| Num layers | 2 – 6 | More layers = more capacity but harder to train |
| Weight decay | `1e-5` – `1e-3` | Higher if overfitting |

### Manual Learning Rate Search

Test: `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` — plot val accuracy for each.

### Optuna (Automated Search)

```python
import optuna

def objective(trial):
    lr           = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    dropout      = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size  = trial.suggest_categorical('hidden_size', [64, 128, 256])

    model     = MyModel(hidden_size, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train for a few epochs
    val_acc = train_and_evaluate(model, optimizer, num_epochs=10)
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)
```

---

## 12. DataLoader Optimization

### Key Parameters

```python
DataLoader(
    dataset,
    batch_size=128,          # Larger: better GPU utilization (but more memory)
    shuffle=True,            # Training only
    num_workers=4,           # Parallel data loading; sweet spot usually 2–8
    pin_memory=True,         # Faster CPU→GPU transfer (GPU training only)
    persistent_workers=True, # Reuse workers across epochs
    prefetch_factor=2,       # Prefetch N batches per worker
    drop_last=True,          # Drop incomplete last batch (useful for BatchNorm)
)
```

### Performance Tips

- `pin_memory=True` + `num_workers>=2` gives biggest speedup on GPU training.
- Too many workers can cause memory issues or I/O bottleneck — profile first.
- `persistent_workers=True` reduces worker startup overhead per epoch.
- For tiny datasets, `num_workers=0` may be faster (no process spawning overhead).

---

## 13. TorchVision Utilities

### Built-in Datasets

```python
from torchvision import datasets

datasets.CIFAR10(root, train=True, download=True, transform=t)
datasets.CIFAR100(root, train=True, download=True, transform=t)
datasets.MNIST(root, train=True, download=True, transform=t)
datasets.EMNIST(root, split='letters', train=True, download=True, transform=t)
datasets.ImageFolder(root='data/train/', transform=t)  # Custom folder structure
datasets.FakeData(size=100, image_size=(3,32,32), num_classes=10, transform=t)
```

### Image Utilities

```python
from torchvision.utils import make_grid, save_image
from torchvision.io import decode_image

# Grid of images for visualization
grid = make_grid(batch_images, nrow=8, padding=2, normalize=True)

# Convert tensor → PIL for display
to_pil = transforms.ToPILImage()
pil_img = to_pil(tensor)

# Save tensor as image file
save_image(tensor, 'output.png', normalize=True)

# Detection / segmentation visualization
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
```

### Pretrained Model Inference

```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

with torch.no_grad():
    output = model(input_tensor)               # [1, 1000]
    probs  = torch.softmax(output, dim=1)
    top5   = torch.topk(probs, 5)
    print(top5.indices, top5.values)
```

---

## 14. Advanced Techniques

### Gradient Accumulation

Simulate larger batch sizes when GPU memory is limited:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs.to(device))
    loss = criterion(outputs, labels.to(device)) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for inputs, labels in train_loader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Batch Normalization Behavior

- **`model.train()`** — uses batch statistics (mean/std of current batch)
- **`model.eval()`** — uses running statistics accumulated during training
- Always call `model.eval()` before inference — otherwise BN uses incorrect statistics.

### Model Parameter Count

```python
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total: {total_params:,} | Trainable: {trainable_params:,}')
```

### Reproducibility

```python
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 15. Debugging & Best Practices

### Tensor Shape Debugging

```python
# Print shapes at each stage
print(f'Input: {x.shape}')
x = self.conv1(x); print(f'After conv1: {x.shape}')
x = self.pool(x);  print(f'After pool:  {x.shape}')
```

### Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Linear layer input size mismatch | Print shapes; use dummy input trick |
| `RuntimeError: Expected all tensors to be on the same device` | CPU/GPU mismatch | Move all tensors to same device |
| `nan` or `inf` loss | Exploding gradients or bad LR | Clip gradients: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` or use gradient accumulation |
| Model not learning | Gradients not flowing | Check `requires_grad`, check for `detach()` misuse |

### Checklist Before Training

- [ ] `torch.manual_seed()` set for reproducibility
- [ ] Data normalized with correct mean/std
- [ ] Model moved to device with `.to(device)`
- [ ] Inputs and labels moved to same device in training loop
- [ ] `optimizer.zero_grad()` at start of each batch
- [ ] `model.train()` before training loop, `model.eval()` before validation
- [ ] `torch.no_grad()` during evaluation
- [ ] Labels in correct format (class indices for CrossEntropyLoss, not one-hot)
- [ ] Scheduler stepped at the right point in the loop

### When to Use `model.eval()`

- During validation / test evaluation
- During inference / prediction
- When computing metrics

This disables:
- Dropout (uses all neurons)
- BatchNorm (uses running stats, not batch stats)

---

## 16. Quick Reference Cheatsheet

### The Complete Training Recipe

```python
# 1. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
dataset = MyDataset(root, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# 3. Model
model = MyModel(num_classes).to(device)

# 4. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# 5. Train
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_acc = evaluate(model, val_loader)
    scheduler.step(val_acc)

# 6. Save
torch.save(model.state_dict(), 'model.pth')

# 7. Load & Infer
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
with torch.no_grad():
    pred = model(input.to(device))
```

### Loss Function Selector

```
Regression          → nn.MSELoss() or nn.L1Loss()
Binary classification → nn.BCEWithLogitsLoss()
Multi-class         → nn.CrossEntropyLoss()   ← most common
```

### Optimizer Selector

```
Fast convergence         → Adam (lr=1e-3)
Fine-tuning pretrained   → SGD (lr=1e-4, momentum=0.9) or Adam (lr=1e-5)
Transformers             → AdamW (lr=1e-4, weight_decay=1e-2)
```

### Scheduler Selector

```
Unknown schedule         → ReduceLROnPlateau (safest default)
Known num epochs         → CosineAnnealingLR
Simple decay             → StepLR
```
