# PyTorch Deep Learning — Comprehensive Course Notes (Expanded)

> Synthesized from:
> - Course 14: PyTorch for Deep Learning (C1–C4)
> - Course 15: PyTorch Techniques and Ecosystem Tools
>
> This is the **expanded version** with explanations, benefits, and annotated code snippets.
> See `pytorch_course_notes.md` for the concise reference version.

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

**What they are:** Tensors are the core data structure in PyTorch — essentially n-dimensional arrays (like NumPy arrays), but with two critical additions: they can live on the GPU for accelerated computation, and they support **autograd** (automatic differentiation), which is how backpropagation works.

**Why they matter:** Everything in PyTorch — images, weights, gradients, predictions — is a tensor. Understanding how to create, reshape, and operate on tensors is the prerequisite for everything else.

### Creation

```python
import torch

# --- From Python data ---
# dtype=torch.float32 is the default for most neural network operations.
# Use float32 (not float64) unless you have a specific reason — it's faster on GPU.
t = torch.tensor([1.0, 2.0, 3.0])
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# --- From NumPy ---
# Shares memory with the numpy array — changes to one affect the other.
import numpy as np
t = torch.from_numpy(np.array([1, 2, 3]))

# --- Predefined shapes ---
# Commonly used to initialize placeholders (e.g., accumulators for loss/metrics).
torch.zeros(3, 4)            # All zeros — good for initializing accumulators
torch.ones(3, 4)             # All ones
torch.rand(3, 4)             # Uniform random [0, 1) — use for weight init testing
torch.randn(3, 4)            # Standard normal — use for dummy inputs to test model shapes
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8] — like Python range(), but a tensor
torch.eye(3)                 # 3×3 identity matrix
```

### Reshaping

**Why this matters:** Neural networks are very strict about tensor shapes. You will constantly need to add/remove/swap dimensions to match what a layer expects.

```python
t.shape                      # Returns a torch.Size object — always check this when debugging

t.reshape(2, 3)              # Rearrange elements into new shape (returns a view if possible)
t.view(2, -1)                # Like reshape, but REQUIRES contiguous memory; -1 = infer this dim

# unsqueeze / squeeze — add or remove a dimension of size 1
# Most common use: adding a batch dimension before passing a single image to a model
t.unsqueeze(0)               # [C, H, W] → [1, C, H, W]  (adds batch dim at front)
t.squeeze()                  # Removes ALL dims of size 1
t.squeeze(0)                 # Removes dim 0 only if it's size 1

# Dimension reordering — critical for image data
# PIL/NumPy images are HWC (Height, Width, Channels)
# PyTorch expects CHW (Channels, Height, Width)
t.transpose(0, 1)            # Swap two specific dimensions
t.permute(2, 0, 1)           # Arbitrary reorder: HWC → CHW (used in ToTensor transform)

t.flatten()                  # Collapse all dims to 1D — used before a Linear layer
t.contiguous()               # Copy data into contiguous memory — needed before .view()
```

**Tip:** `unsqueeze(0)` is the most common reshape operation at inference time — convert `[C, H, W]` → `[1, C, H, W]` to add the batch dimension a model expects.

### Indexing & Slicing

```python
t[1]                         # Select entire row 1
t[:, 2]                      # Select column 2 from ALL rows (: = all)
t[0:2, 1:]                   # Rows 0–1, columns from 1 onward
t[t > 5]                     # Boolean mask — returns a 1D tensor of matching elements
t[[0, 2], :]                 # Fancy indexing — select rows 0 and 2 specifically
t[0, 1].item()               # Extract a single value as a Python float/int (not a tensor)
                             # Use .item() when logging loss values or computing accuracy
```

### Operations

```python
# --- Element-wise arithmetic ---
# Both tensors must have the same shape (or be broadcastable)
a + b, a - b, a * b, a / b

# --- Matrix multiplication ---
# This is the core operation inside Linear layers
torch.matmul(a, b)           # General matrix multiply; also written as a @ b
                             # Shape rule: (M, K) @ (K, N) → (M, N)

# --- Reduction statistics ---
t.mean(), t.std(), t.sum(), t.max(), t.min()
t.mean(dim=0)                # Reduce along dim 0 (collapse rows); result has shape [N_cols]
                             # dim=1 collapses columns; used often in loss/metric calculations

# --- Type casting ---
# Model weights must be float32; labels must be int64 (long) for CrossEntropyLoss
t.float()                    # → float32
t.int()                      # → int32
t.long()                     # → int64 (required for classification labels)
t.bool()                     # → boolean tensor

# --- Comparison & logical ---
# Results are boolean tensors; used for masking and accuracy calculation
a == b, a > b, a < b
(mask1) & (mask2)            # Element-wise AND
(mask1) | (mask2)            # Element-wise OR

# --- Concatenation / stacking ---
torch.cat([a, b], dim=1)     # Join along existing dim — shapes must match on other dims
torch.stack([a, b], dim=0)   # Join along NEW dim — all tensors must have identical shape
```

### Device Management

**What it is:** PyTorch can run computations on the CPU or on a GPU. You have to explicitly move tensors and models to the same device — they can't interact across devices.

**Why it matters:** GPU training can be 10–100× faster for large models. The `device` pattern below is the standard way to write device-agnostic code that works on any machine.

```python
# Auto-detect: use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move a tensor to the target device (returns a new tensor on that device)
t = t.to(device)

# Move the model's weights to the target device
model = model.to(device)

# In the training loop — move EVERY batch to device before passing to the model
inputs, labels = inputs.to(device), labels.to(device)
```

**Tip:** Check device with `t.device`. Mixing CPU and GPU tensors in the same operation causes a runtime error — one of the most common mistakes.

---

## 2. Data Pipeline

**What it is:** The standard pipeline for feeding data into a model: **Dataset → DataLoader → Transforms**.
- `Dataset` defines *how to load one sample*.
- `DataLoader` handles *batching, shuffling, and parallelism*.
- `Transforms` define *how to preprocess each sample*.

**Why this structure:** It decouples data storage from data loading from preprocessing. Each piece is independently swappable, and the DataLoader efficiently parallelizes loading so the GPU is never starved waiting for data.

### Custom Dataset

**When to use:** Any time your data isn't in a standard format (not MNIST, CIFAR, etc.). The three required methods form the contract that `DataLoader` uses.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Build an index of all samples upfront — don't load data here, just paths/labels
        self.samples = self._load_samples()   # Returns list of (image_path, label)

    def __len__(self):
        # DataLoader calls this to know how many samples exist
        return len(self.samples)

    def __getitem__(self, idx):
        # DataLoader calls this for each index in a batch
        # Load ONE sample — keep this fast; avoid slow operations here
        image, label = self._load_item(idx)
        if self.transform:
            image = self.transform(image)    # Apply preprocessing/augmentation
        return image, label                  # Always return (input, target) tuple
```

**Key rules:**
- `__len__` must return the total number of samples.
- `__getitem__` must return a `(sample, label)` tuple.
- Labels from MATLAB/R files are often 1-indexed — subtract 1 to make them 0-indexed for Python.
- Do heavy indexing/setup in `__init__`, not in `__getitem__` (which is called thousands of times).

### Robust Dataset (Error Handling)

**Why needed:** Real-world image datasets often have corrupted files, truncated images, or wrong sizes. A robust dataset skips bad samples instead of crashing the entire training run.

```python
def __getitem__(self, idx):
    # Try the requested index; if it fails, move to the next one
    for _ in range(len(self)):
        try:
            image = Image.open(self.paths[idx]).convert('RGB')
            image.verify()                       # Checks file integrity (detects truncation)
            image = Image.open(self.paths[idx])  # Must re-open after verify() (it closes the file)
            if min(image.size) < 32:
                raise ValueError('Image too small')   # Skip tiny/degenerate images
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f'Skipping {idx}: {e}')
            idx = (idx + 1) % len(self)          # Wrap around to next valid index
```

### DataLoader

**What it is:** Wraps a `Dataset` and provides batching, shuffling, and parallel loading.

**Why it matters:** Training one sample at a time is slow and gives noisy gradient estimates. Batching amortizes the per-sample overhead. Parallel loading (`num_workers`) ensures the GPU processes one batch while the CPU is already loading the next.

```python
from torch.utils.data import DataLoader, random_split

# --- Split a dataset into train / val / test ---
# random_split guarantees no overlap between splits
train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size   # Avoids rounding errors
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Training loader: shuffle so the model doesn't memorize order
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)

# Val/test loaders: no shuffle — order doesn't matter, and it's cleaner for debugging
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
```

**`shuffle=True`** only on training. Never shuffle validation/test — it doesn't change results but is misleading.

### Applying Different Transforms per Split

**Why needed:** `random_split()` returns `Subset` objects that share the parent dataset's transform. If you want augmentation on training but not validation, you need this wrapper pattern.

```python
class SubsetWithTransform(Dataset):
    """Apply a specific transform to a Subset (which has no transform of its own)."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]       # Get the raw (un-transformed) sample
        return self.transform(image), label   # Apply this split's specific transform

# Result: train gets augmentation, val gets only resize+normalize
train_dataset = SubsetWithTransform(train_set, train_transform)
val_dataset   = SubsetWithTransform(val_set,   val_transform)
```

### Transforms

**What they are:** A chain of preprocessing operations applied to each image before it enters the model.

**Why augmentation:** By applying random crops, flips, and color changes at training time, each epoch the model sees slightly different versions of the same image. This acts like having more data and makes the model more robust to real-world variation.

**Critical rule:** Apply augmentation **only to training data**. Val/test must be deterministic so metrics are comparable across runs.

```python
from torchvision import transforms

# --- Training transform (with augmentation) ---
train_transform = transforms.Compose([
    # Step 1 — Geometric transforms (on PIL image, before tensor conversion)
    transforms.RandomResizedCrop(224),       # Random crop + resize: forces model to learn from different scales
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance mirror: doubles effective dataset size for symmetric objects
    transforms.RandomVerticalFlip(p=0.5),   # Only use if vertical flips make sense (e.g., satellite imagery)
    transforms.RandomRotation(degrees=15),  # ±15° rotation: handles tilted inputs

    # Step 2 — Color augmentation
    transforms.ColorJitter(
        brightness=0.2,   # Random brightness shift ±20%: handles different lighting
        contrast=0.2,     # Random contrast shift: handles hazy/sharp images
        saturation=0.2,   # Random saturation: handles faded/vivid images
    ),

    # Step 3 — Convert PIL image to tensor
    # ToTensor does two things: scales [0,255] → [0.0,1.0], and changes HWC → CHW
    transforms.ToTensor(),

    # Step 4 — Normalize each channel to zero mean and unit std
    # These are ImageNet statistics. Use your own dataset's stats if not using a pretrained model.
    # Normalization ensures all channels are on the same scale — speeds up convergence.
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Validation/test transform (no augmentation — fully deterministic) ---
val_transform = transforms.Compose([
    transforms.Resize(256),          # Resize shorter edge to 256, keeping aspect ratio
    transforms.CenterCrop(224),      # Take the center 224×224 patch — same region every time
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Order matters — always follow this sequence:**
1. Geometric transforms (Resize, Crop, Flip, Rotate) — operate on PIL images, faster before tensor
2. Color augmentations (ColorJitter) — also on PIL
3. `ToTensor()` — converts to tensor, scales to [0, 1]
4. `Normalize()` — zero-centers and scales by std

### Computing Dataset Mean & Std

**Why:** Normalization is most effective when you use the actual statistics of your dataset, not generic ImageNet stats. The 2-pass algorithm computes these without loading everything into memory at once.

```python
def compute_mean_std(loader):
    """Compute per-channel mean and std over the entire dataset."""
    mean  = torch.zeros(3)   # One value per channel (R, G, B)
    std   = torch.zeros(3)
    total = 0
    for images, _ in loader:
        batch = images.size(0)
        # Flatten H×W into a single dimension → [B, C, H*W]
        images = images.view(batch, images.size(1), -1)
        mean += images.mean(2).sum(0)   # Sum batch means per channel
        std  += images.std(2).sum(0)    # Sum batch stds per channel
        total += batch
    return mean / total, std / total    # Average over all batches
```

**Tip:** For grayscale datasets used with pretrained RGB models: `transforms.Grayscale(num_output_channels=3)` duplicates the single channel into 3 channels, making the image compatible with pretrained weights.

---

## 3. Model Architecture

**What it is:** The structure of the neural network — which layers are stacked in what order, with what sizes.

**Two patterns for defining models:**

### `nn.Sequential` (Simple)

**Best for:** Quick prototyping or simple linear stacks where you don't need custom logic.

**Limitation:** No flexibility — can't branch, can't have skip connections, can't print intermediate shapes easily.

```python
import torch.nn as nn

# Reads top to bottom: input → Linear → ReLU → Linear → ReLU → output
model = nn.Sequential(
    nn.Linear(4, 64),    # 4 input features → 64 hidden units
    nn.ReLU(),           # Non-linearity — without this, stacking Linears is equivalent to one Linear
    nn.Linear(64, 32),   # 64 → 32
    nn.ReLU(),
    nn.Linear(32, 1)     # 32 → 1 output (regression) or num_classes (classification)
)
```

### `nn.Module` (Recommended)

**Best for:** Any real model. Lets you define custom forward logic, reuse layers, add skip connections, and debug intermediate activations.

**Benefit:** Full control over the computation graph. You can print shapes, add assertions, conditionally apply layers, etc.

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()   # Required — initializes internal PyTorch bookkeeping
        # Define all layers here — PyTorch auto-discovers them as parameters
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.bn1     = nn.BatchNorm1d(hidden_size)   # Normalize activations after fc1
        self.dropout = nn.Dropout(p=0.5)             # Randomly zero 50% of neurons during training
        self.fc2     = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # This is called when you do model(inputs)
        # Read left to right: fc1 → bn1 → relu → dropout → fc2
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        return self.fc2(x)   # Return raw logits — do NOT apply softmax here
```

### Common Layers

| Layer | What it does | When to use |
|---|---|---|
| `nn.Linear(in, out)` | Weighted sum of inputs + bias | Every fully-connected / dense layer |
| `nn.Conv2d(in_ch, out_ch, kernel, padding)` | Learnable filter sliding over 2D image | Any image/spatial data |
| `nn.MaxPool2d(k, stride)` | Keep max value in each k×k window | Downsample feature maps, build translation invariance |
| `nn.AvgPool2d(k)` | Average value in each k×k window | Smoother downsampling; used at the end of CNNs (Global Avg Pool) |
| `nn.BatchNorm1d/2d(features)` | Normalize activations per batch | After Conv/Linear, before activation — stabilizes and speeds up training |
| `nn.Dropout(p)` | Randomly zero p% of neurons | After activations — prevents overfitting; disabled during eval |
| `nn.Flatten()` | Reshape `[B, C, H, W]` → `[B, C*H*W]` | Bridge between conv blocks and linear layers |
| `nn.ReLU()` | `max(0, x)` — zeroes negatives | Default activation; `inplace=True` saves a small amount of memory |
| `nn.GELU()` | Smoother version of ReLU | Used in transformers (BERT, GPT) — slightly better than ReLU for deep networks |
| `nn.Embedding(vocab, dim)` | Lookup table: token index → dense vector | NLP tasks — maps word indices to learned representations |

### CNN Building Pattern

**What CNNs do:** Convolutional layers learn *local* features (edges, textures, shapes). Each block extracts richer features while the spatial resolution shrinks. The final linear layers use these features to classify.

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # --- Block 1: Learn basic edges and textures ---
            # 3 input channels (RGB), 32 output filters, 3×3 kernel
            # padding=1 with kernel=3 preserves spatial size (32×32 stays 32×32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),      # Normalize 32 feature maps — stabilizes early training
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 32×32 → 16×16 (halves H and W)

            # --- Block 2: Learn patterns made of edges (corners, curves) ---
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 16×16 → 8×8

            # --- Block 3: Learn complex object parts ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # 8×8 → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            # 128 × 4 × 4 = 2048 values per sample
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),         # Drop half of neurons — prevents memorizing training data
            nn.Linear(512, num_classes)  # Final output: one score per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**Tip — Calculate classifier input size without doing the math manually:**
```python
# Pass a random tensor through just the feature extractor to see what comes out
dummy = torch.randn(1, 3, 32, 32)            # Batch of 1, 3 channels, 32×32 image
flat_size = model.features(dummy).flatten().shape[0]   # Total elements after features
# Use flat_size as the in_features for the first Linear layer
```

---

## 4. Training Loop

**What it is:** The iterative process of: feed data → compute prediction → measure error → adjust weights. Repeated thousands of times until the model converges.

**Why the exact order matters:** PyTorch accumulates gradients. If you forget `zero_grad()`, gradients from the previous batch add to the current batch's gradients, producing incorrect updates.

### Standard Pattern

```python
import torch
import torch.nn as nn
from torch import optim

# --- Setup (done once before training) ---
model     = MyModel(...).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

train_losses, val_losses, val_accs = [], [], []

for epoch in range(num_epochs):
    # ======= TRAINING PHASE =======
    model.train()   # Activates Dropout and BatchNorm's batch-statistics mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move batch to GPU

        optimizer.zero_grad()               # 1. Clear gradients from previous batch
        outputs = model(inputs)             # 2. Forward pass: compute predictions
        loss = criterion(outputs, labels)   # 3. Compute loss (how wrong are we?)
        loss.backward()                     # 4. Backprop: compute gradients for each weight
        optimizer.step()                    # 5. Update weights using gradients

        # Accumulate loss weighted by batch size (handles variable-size last batch)
        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)

    # ======= VALIDATION PHASE =======
    model.eval()    # Deactivates Dropout; switches BatchNorm to use running statistics
    val_loss = 0.0
    correct  = 0

    with torch.no_grad():   # Disable gradient computation — saves memory, speeds up eval
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = outputs.max(1)                         # Predicted class = highest logit
            correct  += (preds == labels).sum().item()        # Count correct predictions

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc  = correct  / len(val_loader.dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    # Step scheduler AFTER validation — it needs the metric to decide whether to reduce LR
    scheduler.step(epoch_val_acc)

    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} | '
          f'Val Acc: {epoch_val_acc:.4f} | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
```

### Key Training Steps — Order Matters

1. `optimizer.zero_grad()` — clear previous gradients *(forget to do this and gradients accumulate)*
2. `outputs = model(inputs)` — forward pass through the network
3. `loss = criterion(outputs, labels)` — measure how wrong the predictions are
4. `loss.backward()` — backpropagate: compute how much each weight contributed to the error
5. `optimizer.step()` — adjust each weight in the direction that reduces error

**Never forget `zero_grad()`!** PyTorch accumulates gradients by default, which is useful for some advanced techniques (like gradient accumulation), but wrong for a standard training loop.

### Saving & Loading Checkpoints

**Why:** Save the best model during training so you can restore it after early stopping, or after a crash. Always save the optimizer state too — if you resume training, the optimizer needs its internal state (momentum buffers, adaptive LR estimates) to continue correctly.

```python
# --- Save when validation improves ---
torch.save({
    'epoch'              : epoch,
    'model_state_dict'   : model.state_dict(),      # All weights and biases
    'optimizer_state_dict': optimizer.state_dict(), # Optimizer internal state (e.g., Adam moments)
    'val_loss'           : best_val_loss,
}, 'checkpoint.pth')

# --- Load checkpoint ---
checkpoint = torch.load('checkpoint.pth', map_location=device)  # map_location handles GPU→CPU migration
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 5. Loss Functions & Optimizers

### Loss Functions

**What they are:** A function that measures how wrong the model's predictions are, producing a single scalar number. The training loop minimizes this number.

**Why choosing the right one matters:** Using the wrong loss function (e.g., MSE for classification) leads to incorrect gradient signals and poor learning. Each loss is mathematically tailored to its task.

| Task | Loss | Why this one |
|---|---|---|
| Regression (continuous output) | `nn.MSELoss()` | Penalizes errors quadratically — large errors are punished heavily |
| Regression (outlier-robust) | `nn.L1Loss()` | Penalizes errors linearly — less sensitive to outliers than MSE |
| Binary classification (yes/no) | `nn.BCEWithLogitsLoss()` | Numerically stable sigmoid + binary cross-entropy in one step |
| Multi-class classification | `nn.CrossEntropyLoss()` | Combines log-softmax + negative log-likelihood — standard for classification |

```python
# --- MSELoss: regression ---
# Penalizes (prediction - target)² — large errors are punished more than small ones
# Use when: predicting house prices, temperatures, any continuous value
criterion = nn.MSELoss()
loss = criterion(predicted_price, actual_price)   # Both are float tensors

# --- L1Loss: robust regression ---
# Penalizes |prediction - target| — same penalty regardless of error magnitude
# Use when: your targets have outliers (e.g., salary data with a few billionaires)
criterion = nn.L1Loss()

# --- BCEWithLogitsLoss: binary classification ---
# Use when: spam/not-spam, dog/not-dog, sick/healthy — exactly TWO classes
# Combines sigmoid (squishes logit to [0,1]) + binary cross-entropy
# IMPORTANT: pass raw logits (no sigmoid before), labels must be float (0.0 or 1.0)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logit, label.float())

# --- CrossEntropyLoss: multi-class classification ---
# Use when: 3+ classes (cat/dog/bird, digits 0-9, etc.)
# Internally applies log-softmax then NLLLoss
# IMPORTANT: pass raw logits (no softmax before); labels must be class INDICES (not one-hot)
criterion = nn.CrossEntropyLoss()
outputs = model(inputs)               # Shape: [batch_size, num_classes] — raw logits
loss    = criterion(outputs, labels)  # labels: [batch_size] with integers 0..num_classes-1
```

### Optimizers

**What they are:** Algorithms that update model weights using the gradients computed by `loss.backward()`. Different optimizers use different strategies for how to apply gradient updates.

**Why it matters:** The choice of optimizer and learning rate is the single biggest factor (after architecture) in how fast and how well a model trains.

```python
# --- SGD (Stochastic Gradient Descent) ---
# The simplest optimizer: weight -= lr * gradient
# momentum=0.9 adds a "momentum" term that smooths updates and escapes local minima
# weight_decay applies L2 regularization (penalizes large weights)
# Use when: fine-tuning pretrained models (proven to generalize better), or when you want
#           more control over the learning dynamics
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# --- Adam (Adaptive Moment Estimation) ---
# Maintains a separate adaptive learning rate for each parameter
# Converges faster than SGD, less sensitive to the initial LR choice
# Use when: training from scratch, prototyping, most general-purpose use
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --- AdamW (Adam with decoupled weight decay) ---
# Fixes a subtle bug in Adam where weight decay interacts with adaptive LR scaling
# Practically: better regularization than Adam
# Use when: training transformers (BERT, GPT), or any large modern architecture
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

**Fine-tuning tip:** Use a much lower learning rate (`1e-5` to `1e-4`) when fine-tuning pretrained models. High LR will destroy the carefully learned ImageNet features before the new head is ready.

### Weight Decay (L2 Regularization)

**What it is:** An additional term added to the loss that penalizes large weight values, pushing weights toward zero.

**Why it helps:** Large weights mean the model is very sensitive to small changes in input (i.e., it's over-fitted). Weight decay forces the model to use smaller, more spread-out weights — it learns a simpler, more generalizable function. Typical value: `1e-4`.

---

## 6. Learning Rate Scheduling

**What it is:** Automatically adjusting the learning rate during training according to a schedule — starting high and reducing it over time (or when progress stalls).

**Why it helps:** A high learning rate at the start lets the model take big steps toward a good region. Once near a good solution, a large LR causes the model to "bounce around" and never converge. Reducing the LR later allows fine-grained refinement. Think of it like zooming in: coarse search first, then precision.

**Key rule:** Always call `scheduler.step()` *after* the optimizer step, and *after* validation (for plateau-based schedulers).

### StepLR

**What it does:** Multiplies the LR by `gamma` every `step_size` epochs. Simple and predictable.

**Use when:** You have a rough idea of how many epochs your training will run and want a predetermined decay schedule.

```python
# Reduce LR by factor of 0.2 every 10 epochs
# e.g., start: 0.01 → epoch 10: 0.002 → epoch 20: 0.0004
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

# Call after each epoch (not after each batch)
scheduler.step()
```

### CosineAnnealingLR

**What it does:** Smoothly decays the LR following a cosine curve from the initial value down to `eta_min` over `T_max` epochs. No sudden drops — the LR changes continuously.

**Use when:** You know the total number of training epochs and want smooth, gradual decay. Often produces better final accuracy than StepLR.

```python
# LR follows cosine curve: starts at initial LR, ends at eta_min after T_max epochs
# The cosine shape means LR decays slowly at first, faster in the middle, slowly at the end
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,   # Total epochs over which to complete one cosine cycle
    eta_min=1e-6        # Minimum LR — prevents LR from reaching zero
)
scheduler.step()        # Call after each epoch
```

### ReduceLROnPlateau (Recommended Default)

**What it does:** Monitors a metric (e.g., val accuracy). If it doesn't improve for `patience` epochs, the LR is multiplied by `factor`. Fully adaptive — no need to know in advance how long to train.

**Use when:** You're unsure about the training duration, or you want the model to automatically adapt. This is the safest default for most experiments.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # 'max' because we monitor accuracy (higher is better)
                     # Use 'min' if monitoring loss (lower is better)
    factor=0.5,      # Multiply LR by 0.5 when triggered (e.g., 1e-3 → 5e-4)
    patience=3,      # Wait 3 epochs without improvement before reducing
    verbose=True,    # Print a message when LR is reduced
)

# For ReduceLROnPlateau only: pass the metric value — the scheduler decides whether to reduce
scheduler.step(val_accuracy)   # Called AFTER validation, not after each batch
```

### Get Current LR

```python
# Always check the current LR when debugging training issues
current_lr = optimizer.param_groups[0]['lr']
print(f'Current LR: {current_lr:.2e}')
```

---

## 7. Evaluation & Metrics

**What they are:** Quantitative measures of how well the model performs on unseen data. Computed after training is complete, or after each epoch on the validation set.

**Why multiple metrics:** Accuracy alone is misleading on imbalanced datasets. If 90% of your data is class A and 10% is class B, a model that always predicts class A achieves 90% accuracy while being completely useless for class B. Precision, recall, and F1 expose this.

### Accuracy (Manual)

**What it is:** Fraction of predictions that are correct. Simple and intuitive.

**When to use:** Balanced datasets where all classes appear roughly equally.

```python
model.eval()    # MUST switch to eval mode before any evaluation
correct = total = 0

with torch.no_grad():   # Disable autograd — no gradients needed for inference
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)                        # Raw logits: [B, num_classes]
        _, predicted = outputs.max(1)                  # max returns (values, indices); we want indices
        correct += (predicted == labels).sum().item()  # Count matches; .item() converts tensor to Python int
        total   += labels.size(0)                      # Accumulate total samples

accuracy = correct / total   # Proportion correct
```

### TorchMetrics (Recommended)

**What it is:** A library that provides robust, GPU-compatible metric implementations. Handles edge cases (empty batches, distributed training) that manual implementations miss.

**Benefit:** Metrics accumulate state across batches automatically, then compute the final value with `.compute()`. Cleaner and less error-prone than manual accumulation.

```python
import torchmetrics

# Move metrics to the same device as the model — they store internal state tensors
accuracy  = torchmetrics.Accuracy(task='multiclass', num_classes=N, average='micro').to(device)
precision = torchmetrics.Precision(task='multiclass', num_classes=N, average='macro').to(device)
recall    = torchmetrics.Recall(task='multiclass', num_classes=N, average='macro').to(device)
f1        = torchmetrics.F1Score(task='multiclass', num_classes=N, average='macro').to(device)

# In evaluation loop — update() accumulates predictions without computing yet
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        preds   = outputs.argmax(dim=1)          # Convert logits to predicted class index
        accuracy.update(preds, labels.to(device))
        f1.update(preds, labels.to(device))

# After iterating all batches — compute() returns the final metric value
print(f'Accuracy: {accuracy.compute():.4f}')
print(f'F1:       {f1.compute():.4f}')

accuracy.reset()   # Reset internal state before next epoch — IMPORTANT, or values accumulate across epochs
```

### Averaging Modes Explained

**Why different modes exist:** In multi-class problems, you need to decide how to aggregate per-class performance into a single number. Each mode tells a different story.

| Mode | What it computes | Use when |
|---|---|---|
| `micro` | Treats all samples equally (equivalent to standard accuracy) | Balanced datasets — this is just accuracy |
| `macro` | Computes metric per class, then averages (each class equally weighted) | Imbalanced datasets — forces the model to perform well on rare classes |
| `weighted` | Computes metric per class, weighted by class frequency | When you care about overall performance but acknowledge class imbalance |

**Example:** If class A has 900 samples (99% correct) and class B has 100 samples (10% correct):
- `micro` accuracy = 90.1% ← misleadingly high
- `macro` accuracy = (99% + 10%) / 2 = 54.5% ← reveals the failure on class B

### Plotting Loss & Accuracy

**Why:** Training curves are the primary diagnostic tool. A growing gap between train and val loss means overfitting. A val loss that never improves means underfitting or a bad LR.

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# --- Loss curve: look for the point where val loss diverges from train loss ---
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
# If val loss starts increasing while train loss keeps falling: overfitting

# --- Accuracy curve: val accuracy should track train accuracy closely ---
ax2.plot(val_accs, label='Val Accuracy', color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')   # Save high-res version
plt.show()
```

---

## 8. Convolutional Neural Networks (CNNs)

**What they are:** Neural networks specialized for grid-structured data (images, audio spectrograms). Instead of connecting every input pixel to every neuron (which would be billions of parameters for a 224×224 image), CNNs use small learnable filters that slide across the image.

**Why they work:** The same filter learns to detect the same feature (e.g., a vertical edge) everywhere in the image, regardless of position. This is called **translation equivariance** and is why CNNs need far fewer parameters than fully-connected networks for images.

### Core Building Blocks

| Block | What it does | Why you need it |
|---|---|---|
| `Conv2d` | Learnable filter sliding across image | Extracts local features: edges → textures → shapes → objects |
| `BatchNorm2d` | Normalizes activations per batch | Prevents gradients from vanishing/exploding; allows higher LR; acts as regularizer |
| `ReLU` | `max(0, x)` — zeroes negatives | Introduces non-linearity; without it, stacked Conv layers would be equivalent to one |
| `MaxPool2d` | Keeps max value in each window | Reduces spatial size (less computation); builds translation invariance |
| `Dropout` | Randomly zeros a fraction of neurons | Prevents co-adaptation; forces the network to learn redundant representations |
| `Flatten` | Collapses spatial dims to 1D vector | Bridge from conv (2D) to linear (1D) layers |

### Spatial Dimension Formula

**Why you need this:** When you stack Conv and Pool layers, the spatial size changes. You need to know the size to set the correct `in_features` for the first Linear layer.

```
output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1

With kernel=3, padding=1, stride=1:  output = input  (size preserved — most common CNN setting)
With MaxPool2d(kernel=2, stride=2):  output = input / 2  (halves the spatial dimensions)
```

**Example trace for a 32×32 input through 3 Conv+Pool blocks:**
- Input: 32×32 → after Block 1 MaxPool: 16×16 → after Block 2 MaxPool: 8×8 → after Block 3 MaxPool: 4×4
- After Flatten: 4 × 4 × 128 = 2048 features

### Overfitting Diagnosis

**What overfitting is:** The model memorizes training data instead of learning generalizable patterns. It performs well on training data but poorly on new data.

**How to detect it:**
- Training loss keeps decreasing
- Validation loss stops decreasing or starts increasing
- Large gap opens up between train and val accuracy

**How to fix it (ordered by how easy they are to apply):**
- Add `Dropout` (p=0.3–0.5) — easiest, try this first
- Add data augmentation — free extra training data
- Add `BatchNorm` — regularizes activations
- Add `weight_decay` to optimizer — L2 penalty on weights
- Reduce model capacity (fewer layers/channels) — simpler model = less memorization
- Use early stopping — stop training when val loss stops improving

---

## 9. Transfer Learning & Fine-Tuning

**What it is:** Starting with a model already trained on a large dataset (typically ImageNet with 1.2M images and 1000 classes), then adapting it for your specific task.

**Why it works:** The lower layers of a CNN learn general features — edges, corners, color gradients, textures — that are useful for almost any image task. You don't need to re-learn these from scratch. You only need to adapt the final layers for your specific classes.

**When to use:** Almost always. Even if your task is very different from ImageNet, pretrained weights provide a much better starting point than random initialization. Transfer learning typically reaches better accuracy with less data and less training time.

### Load a Pretrained Model

```python
import torchvision.models as models

# Modern API — always specify weights explicitly (avoids deprecation warnings)
# IMAGENET1K_V1 = trained on ImageNet with 1000 classes
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
```

### Replace the Classifier Head

**Why:** The pretrained model's final layer outputs 1000 class scores (ImageNet). You need to replace it with a layer that outputs your number of classes.

```python
# --- ResNet-style: single fc (fully-connected) layer at the end ---
in_features = model.fc.in_features     # Get the input size of the existing head
model.fc = nn.Linear(in_features, num_classes)   # Replace with your head

# --- MobileNet/EfficientNet-style: Sequential classifier block at the end ---
in_features = model.classifier[-1].in_features   # Get input size of last layer in block
model.classifier[-1] = nn.Linear(in_features, num_classes)   # Replace only the last layer
```

### Strategy 1 — Feature Extraction (Fastest)

**What it does:** Freezes ALL backbone weights. Only the new head is trained. The pretrained CNN is used as a fixed feature extractor.

**Best for:** Small datasets (< a few thousand images), or when your domain is very similar to ImageNet (natural images). Fast because very few parameters are updated.

```python
# Freeze everything — require_grad=False means no gradient is computed for these
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the new head — these are the only layers that will update
for param in model.fc.parameters():
    param.requires_grad = True

# Only pass parameters that actually require gradients to the optimizer
# filter() ensures frozen parameters don't waste memory on optimizer state
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

### Strategy 2 — Fine-Tuning (Balanced, Two-Stage)

**What it does:** Stage 1 trains only the head (same as Strategy 1). Stage 2 unfreezes the top backbone layers and continues training both with a lower LR for the backbone.

**Why two stages:** If you unfreeze the backbone too early, the randomly initialized head produces large gradients that corrupt the pretrained features. By training the head first, you ensure it's providing sensible gradients before the backbone starts updating.

**Best for:** Medium datasets, most tasks. Often the best accuracy-to-training-time ratio.

```python
# --- Stage 1: Train head only for ~5 epochs (same as feature extraction) ---
# (see Strategy 1 above)

# --- Stage 2: Unfreeze top backbone layers ---
# For ResNet: layer4 is the deepest conv block — closest to the output, most task-specific
for param in model.layer4.parameters():
    param.requires_grad = True

# Use different learning rates per group:
# - backbone LR is 10-100× lower to avoid destroying pretrained features
# - head LR is higher because its weights need more adjustment
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},   # Very low LR for backbone
    {'params': model.fc.parameters(),     'lr': 1e-4},   # Higher LR for head
])
```

### Strategy 3 — Full Retraining

**What it does:** All layers are trainable from the start. The pretrained weights serve only as initialization.

**Best for:** Large datasets (tens of thousands+ of images), or tasks very different from ImageNet (e.g., medical images, satellite imagery). Requires more training time and data to avoid overfitting.

```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Moderate LR — too high will destroy pretrained initialization, too low converges slowly
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### Standard Pretrained Input Preprocessing

**Why:** Pretrained models expect images preprocessed exactly as they were during their original training. Using different normalization statistics will shift all activations and degrade performance.

```python
# These are the exact ImageNet statistics used to train all standard torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

pretrained_transform = transforms.Compose([
    transforms.Resize(256),          # Resize to slightly larger than target
    transforms.CenterCrop(224),      # Crop to 224×224 — standard input size for most models
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

**Tip:** Grayscale to 3-channel for pretrained models: `transforms.Grayscale(num_output_channels=3)`. This duplicates the grayscale channel 3 times, making the tensor 3-channel so the pretrained Conv2d can accept it.

---

## 10. Regularization Techniques

**What regularization is:** Techniques that reduce overfitting by preventing the model from relying too heavily on any single feature or pattern. Overfitting is when the model memorizes training data instead of learning generalizable patterns.

**General principle:** Regularization adds constraints or noise that make the model's job slightly harder during training — forcing it to find more robust solutions.

| Technique | How it works | Benefit | When to apply |
|---|---|---|---|
| **Dropout** | Randomly zeros p% of neurons each forward pass | Forces redundant representations; neurons can't co-adapt | After activations in FC layers; p=0.3–0.5 |
| **Weight Decay (L2)** | Adds penalty term `λ × sum(w²)` to loss | Keeps weights small; simpler model generalizes better | Via `weight_decay` in optimizer; λ=1e-4 |
| **Batch Normalization** | Normalizes activations within each mini-batch | Reduces internal covariate shift; acts as implicit regularizer | After Conv/Linear, before activation |
| **Data Augmentation** | Applies random transforms at training time | Model sees more diverse inputs; builds invariances | Training transform only; not val/test |
| **Early Stopping** | Stops training when val loss stops improving | Prevents training into the overfitting regime | Monitor val loss, stop after N bad epochs |
| **Label Smoothing** | Softens one-hot targets (e.g., 1.0→0.9, 0.0→0.033) | Prevents model from being overconfident; better calibration | Via `CrossEntropyLoss(label_smoothing=0.1)` |

**In practice:** Start with data augmentation + dropout. If still overfitting, add weight decay. If training is unstable, add batch normalization.

---

## 11. Hyperparameter Tuning

**What it is:** Finding the best values for parameters that are not learned by the model (unlike weights, which are learned automatically). These are set before training and control how training happens.

**Why it matters:** The same architecture with a learning rate of `1e-1` might diverge completely while `1e-3` converges perfectly. Hyperparameters can make or break a model.

### Key Hyperparameters and Search Ranges

| Hyperparameter | Typical Range | Why it matters |
|---|---|---|
| Learning rate | `1e-5` – `1e-1` | Most impactful single hyperparameter. Too high = diverge, too low = slow convergence |
| Batch size | 16, 32, 64, 128 | Larger batches give smoother gradient estimates but may hurt generalization; need more memory |
| Dropout rate | 0.1 – 0.5 | Higher values = stronger regularization; tune based on how much the model overfits |
| Hidden units | 64, 128, 256, 512 | More units = more capacity to learn; also more risk of overfitting |
| Num layers | 2 – 6 | More layers = more abstract representations; harder to train (vanishing gradients) |
| Weight decay | `1e-5` – `1e-3` | Higher = more regularization; increase if val loss diverges from train loss |

### Manual Learning Rate Search

**Strategy:** Logarithmic sweep. Test orders of magnitude, not linear increments. Run 3–5 epochs per LR, plot val accuracy — pick the LR where accuracy rises fastest.

```
Test: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] — plot val accuracy for each after a few epochs
The best LR is usually one step before where the loss diverges.
```

### Optuna (Automated Search)

**What it is:** A framework for automated hyperparameter optimization. Instead of trying all combinations (grid search), Optuna uses a probabilistic model (Tree-structured Parzen Estimator) to focus trials on promising regions of the search space.

**Benefit over grid search:** Much more efficient. Grid search with 5 values for 4 hyperparameters = 625 trials. Optuna finds good solutions in 20–50 trials by learning which regions are promising.

```python
import optuna

def objective(trial):
    # trial.suggest_*() samples a value from the search space
    # Optuna tracks which values led to good results and focuses future trials there
    lr          = trial.suggest_float('lr', 1e-5, 1e-1, log=True)   # Log scale: more trials near 1e-3
    dropout     = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])

    model     = MyModel(hidden_size, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train for a few epochs — enough to distinguish good from bad hyperparameters
    val_acc = train_and_evaluate(model, optimizer, num_epochs=10)
    return val_acc   # Optuna maximizes this value

# direction='maximize' for accuracy, 'minimize' for loss
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)   # Run 50 trials

print(study.best_params)   # Best hyperparameter combination found
```

---

## 12. DataLoader Optimization

**What it is:** Configuring the DataLoader to minimize the time the GPU spends waiting for data. If data loading is slow, the GPU sits idle between batches — wasting expensive compute time.

**Why it matters:** For large models on fast GPUs, data loading is often the bottleneck. Proper configuration can increase GPU utilization from 40% to 95%+.

### Key Parameters

```python
DataLoader(
    dataset,
    batch_size=128,           # Larger batches: better GPU utilization and more stable gradients
                              # Constraint: must fit in GPU memory. Start high, reduce if OOM.

    shuffle=True,             # Randomize sample order each epoch — only for training
                              # Prevents the model from learning order-based patterns

    num_workers=4,            # Number of CPU processes loading data in parallel
                              # Rule of thumb: start with 4; sweet spot usually 2–8
                              # Too many: memory/OS overhead; too few: GPU starves

    pin_memory=True,          # Lock loaded data into RAM for faster CPU→GPU transfer
                              # Only enable for GPU training (on CPU it wastes memory)

    persistent_workers=True,  # Keep worker processes alive between epochs
                              # Without this: workers are spawned/killed each epoch (slow)

    prefetch_factor=2,        # Each worker prefetches 2 batches ahead of consumption
                              # GPU processes batch N while CPU loads batch N+2

    drop_last=True,           # Discard the last batch if it's smaller than batch_size
                              # Important for BatchNorm: a batch of size 1 has undefined variance
)
```

### Performance Tips

- `pin_memory=True` + `num_workers >= 2` gives the biggest single speedup for GPU training.
- Too many workers can cause RAM exhaustion or hit OS file descriptor limits — profile before increasing.
- `persistent_workers=True` eliminates the ~1-2 second startup overhead per epoch for processes with expensive initialization.
- For tiny datasets that fit in RAM: preload everything into memory and use `num_workers=0` — process spawning overhead isn't worth it.

---

## 13. TorchVision Utilities

**What it is:** A companion library to PyTorch providing ready-to-use datasets, pretrained models, and image processing utilities — so you don't have to implement them from scratch.

### Built-in Datasets

**Why use them:** Zero setup — download and use in one line. Useful for prototyping and benchmarking your pipeline before applying it to custom data.

```python
from torchvision import datasets

# Standard benchmarks — great for testing that your pipeline works before using real data
datasets.CIFAR10(root, train=True, download=True, transform=t)    # 60K images, 10 classes
datasets.CIFAR100(root, train=True, download=True, transform=t)   # 60K images, 100 classes
datasets.MNIST(root, train=True, download=True, transform=t)      # 70K 28×28 handwritten digits
datasets.EMNIST(root, split='letters', train=True, download=True, transform=t)  # Extended MNIST

# For your own organized images — expects root/class_name/image.jpg structure
# Automatically assigns class labels based on folder names (alphabetically sorted)
datasets.ImageFolder(root='data/train/', transform=t)

# Synthetic data for pipeline testing without downloading anything
# Useful for verifying your model runs end-to-end before real data is ready
datasets.FakeData(size=100, image_size=(3, 32, 32), num_classes=10, transform=t)
```

### Image Utilities

```python
from torchvision.utils import make_grid, save_image

# --- Visualization grid ---
# Arranges a batch of images into a single grid image for quick visual inspection
# normalize=True rescales pixel values to [0,1] for display (reverses normalization)
grid = make_grid(batch_images, nrow=8, padding=2, normalize=True)

# --- Convert tensor → PIL for matplotlib display ---
# PyTorch tensors are CHW float [0,1]; PIL images are HWC uint8 [0,255]
to_pil  = transforms.ToPILImage()
pil_img = to_pil(tensor)

# --- Save tensor directly to disk ---
save_image(tensor, 'output.png', normalize=True)

# --- Annotation utilities (for detection / segmentation models) ---
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
# draw_bounding_boxes(image_uint8, boxes, labels, colors, width)
# draw_segmentation_masks(image_uint8, masks, alpha=0.5)
```

### Pretrained Model Inference

**Standard inference pipeline:** The model must be in `eval()` mode and the input must be preprocessed identically to how the model was trained.

```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()   # Disable Dropout and switch BatchNorm to use running statistics

with torch.no_grad():   # No gradients needed — saves memory and speeds up inference
    output = model(input_tensor)            # [1, 1000] — raw logits for 1000 ImageNet classes
    probs  = torch.softmax(output, dim=1)  # Convert logits to probabilities (sum to 1)
    top5   = torch.topk(probs, 5)         # Get top 5 predictions
    print(top5.indices, top5.values)       # Class indices and their probabilities
```

---

## 14. Advanced Techniques

### Gradient Accumulation

**What it is:** A technique to simulate a larger batch size by accumulating gradients over multiple smaller batches before performing an optimizer step.

**Why needed:** Large batch sizes give more stable gradient estimates, but may not fit in GPU memory. With gradient accumulation of 4 steps, you effectively use 4× the batch size without needing 4× the GPU memory.

```python
accumulation_steps = 4       # Effective batch size = batch_size × accumulation_steps
optimizer.zero_grad()        # Zero gradients at the start of the accumulation window

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs.to(device))
    # Divide loss by accumulation_steps so the scale matches a single large batch
    loss = criterion(outputs, labels.to(device)) / accumulation_steps
    loss.backward()          # Accumulate gradients (don't step yet)

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()     # Now update weights using accumulated gradients
        scheduler.step()
        optimizer.zero_grad()  # Reset for next accumulation window
```

### Mixed Precision Training

**What it is:** Running the forward pass in float16 (half precision) instead of float32. The backward pass and weight updates still use float32 for numerical stability.

**Why use it:** float16 uses half the memory and is 2–8× faster on modern NVIDIA GPUs (Tensor Cores). This lets you use larger batch sizes or larger models on the same hardware.

```python
from torch.cuda.amp import GradScaler, autocast

# GradScaler prevents gradient underflow — float16 has limited range, so very small
# gradients round to zero. The scaler multiplies the loss before backward to amplify them.
scaler = GradScaler()

for inputs, labels in train_loader:
    optimizer.zero_grad()

    # autocast: within this block, eligible ops run in float16 automatically
    with autocast():
        outputs = model(inputs.to(device))
        loss    = criterion(outputs, labels.to(device))

    scaler.scale(loss).backward()   # Scale loss before backward to prevent underflow
    scaler.step(optimizer)          # Unscales gradients and then calls optimizer.step()
    scaler.update()                 # Updates the scale factor for next iteration
```

### Batch Normalization Behavior

**What it is:** BatchNorm normalizes activations within each mini-batch during training. During inference, it uses statistics accumulated over the entire training set.

**Why this matters:** The behavior is fundamentally different in train vs eval mode. If you forget to call `model.eval()`, BatchNorm uses batch statistics at inference — which are unstable for small batches and wrong for single-image inference.

```python
# TRAINING mode (model.train()):
# BatchNorm computes mean/std of the CURRENT BATCH and normalizes with it
# Also updates running_mean and running_std (exponential moving average)

# EVALUATION mode (model.eval()):
# BatchNorm uses running_mean and running_std accumulated during training
# This is deterministic and correct — the same input always gives the same output

# Always call model.eval() before any validation/inference!
model.eval()
with torch.no_grad():
    output = model(input)   # Now BatchNorm behaves correctly for inference
```

### Model Parameter Count

**Why check this:** Parameter count tells you roughly how complex a model is and whether it'll fit in memory. It's also a useful sanity check after replacing a classifier head.

```python
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# numel() = number of elements in the tensor (e.g., a 3×3 Conv has 9 weight elements per channel pair)

print(f'Total:     {total_params:,}')       # All parameters (frozen + trainable)
print(f'Trainable: {trainable_params:,}')   # Only the ones the optimizer will update
```

### Reproducibility

**Why it matters:** Neural network training involves many random operations (weight initialization, dropout, data shuffling). Without fixing seeds, two runs with identical code produce different results, making it impossible to compare experiments or debug issues.

```python
import random
import numpy as np

SEED = 42   # Any fixed integer; 42 is conventional but arbitrary

# Each library has its own random state — must set all of them
torch.manual_seed(SEED)          # PyTorch CPU operations
torch.cuda.manual_seed_all(SEED) # PyTorch GPU operations (all GPUs)
np.random.seed(SEED)             # NumPy (used by many transform libraries)
random.seed(SEED)                # Python's built-in random

# These settings make cuDNN deterministic at a small speed cost
torch.backends.cudnn.deterministic = True   # Use deterministic algorithms
torch.backends.cudnn.benchmark = False      # Disable auto-tuning (it's non-deterministic)
```

---

## 15. Debugging & Best Practices

**Why debugging skills matter:** Deep learning bugs are often silent — the model trains without errors but learns the wrong thing. A good debugger catches issues early before wasting hours of compute.

### Tensor Shape Debugging

**The most common source of bugs** in PyTorch is mismatched tensor shapes. Print shapes at each layer until you find where the expected shape diverges from the actual shape.

```python
# Add temporary print statements in forward() to trace shapes
def forward(self, x):
    print(f'Input:       {x.shape}')    # e.g., [32, 3, 32, 32]
    x = self.conv1(x)
    print(f'After conv1: {x.shape}')   # e.g., [32, 32, 32, 32]
    x = self.pool(x)
    print(f'After pool:  {x.shape}')   # e.g., [32, 32, 16, 16]
    x = self.flatten(x)
    print(f'After flat:  {x.shape}')   # e.g., [32, 8192]
    # The output of flatten's shape[1] must match Linear's in_features
```

### Common Errors and Fixes

| Error | Root cause | Fix |
|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Linear layer in_features doesn't match the actual flattened size | Print shapes; use dummy input trick to compute correct size |
| `RuntimeError: Expected all tensors on the same device` | Model is on GPU but data is on CPU (or vice versa) | Add `.to(device)` to inputs and labels inside the training loop |
| `nan` or `inf` loss | Exploding gradients (LR too high) or log of zero | Clip gradients: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`; reduce LR |
| `CUDA out of memory` | Batch too large, or gradient accumulated across val loop | Reduce batch_size; ensure `torch.no_grad()` wraps validation |
| Model not learning (loss stuck) | Gradients not flowing; bad weight init; wrong LR | Check `requires_grad`; check for accidental `.detach()`; try LR=1e-3 |
| Loss NaN at first batch | Bad data (NaN in input); learning rate way too high | Check `torch.isnan(inputs).any()`; try LR=1e-4 |

### Checklist Before Training

Going through this list before a long training run saves hours of debugging:

- [ ] `torch.manual_seed()` set for reproducibility
- [ ] Data normalized with the correct mean/std (your dataset's, not ImageNet's, if training from scratch)
- [ ] Model moved to device with `.to(device)`
- [ ] Inputs and labels moved to same device inside the training loop
- [ ] `optimizer.zero_grad()` at start of each batch (not end!)
- [ ] `model.train()` before training phase, `model.eval()` before validation
- [ ] `torch.no_grad()` wraps the validation loop
- [ ] Labels are class indices (integers), not one-hot vectors, for CrossEntropyLoss
- [ ] Scheduler stepped at the correct point (after validation for ReduceLROnPlateau)
- [ ] Checkpoint directory exists and has write permissions

### When to Use `model.eval()`

**Always** switch to eval mode before any evaluation or inference. Forgetting this is a common silent bug — the model appears to work but metrics are wrong.

```python
model.eval()   # Switches TWO behaviors:
               # 1. Dropout: disabled — all neurons are active (full model capacity)
               # 2. BatchNorm: uses accumulated running statistics, not batch statistics
               #    (critical for single-image inference where batch stats are meaningless)
```

Switch back with `model.train()` before the next training epoch.

---

## 16. Quick Reference Cheatsheet

### The Complete Training Recipe

```python
# 1. Device — auto-detect GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data — transform, dataset, loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
dataset      = MyDataset(root, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# 3. Model — build and move to device
model = MyModel(num_classes).to(device)

# 4. Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

# 5. Train loop
for epoch in range(num_epochs):
    model.train()                               # ← Activate Dropout + BatchNorm train mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()                   # ← MUST do this first
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()                                # ← Switch to eval mode for validation
    with torch.no_grad():                       # ← No gradients needed for validation
        val_acc = evaluate(model, val_loader)
    scheduler.step(val_acc)                     # ← Step scheduler AFTER validation

# 6. Save best model
torch.save(model.state_dict(), 'model.pth')

# 7. Load and infer
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
with torch.no_grad():
    pred = model(input.unsqueeze(0).to(device))  # unsqueeze(0) adds the batch dimension
```

### Loss Function Decision Guide

```
Output is a continuous number (price, temperature)?   → nn.MSELoss()     (or L1Loss if outliers)
Output is YES or NO (two classes only)?               → nn.BCEWithLogitsLoss()
Output is one of 3+ classes?                          → nn.CrossEntropyLoss()   ← most common

Key rule: CrossEntropyLoss and BCEWithLogitsLoss expect RAW LOGITS.
          Do NOT apply softmax/sigmoid before passing to the loss.
```

### Optimizer Decision Guide

```
Training from scratch, want fast convergence?   → Adam    (lr=1e-3)
Fine-tuning a pretrained model?                 → Adam    (lr=1e-5) or SGD (lr=1e-4, momentum=0.9)
Training a transformer (BERT, GPT)?             → AdamW   (lr=1e-4, weight_decay=1e-2)

Backbone vs head have different LRs?
→ Use parameter groups:
  optimizer = Adam([
      {'params': model.backbone.parameters(), 'lr': 1e-5},  # Low LR for backbone
      {'params': model.head.parameters(),     'lr': 1e-3},  # High LR for head
  ])
```

### Scheduler Decision Guide

```
Don't know how long to train?              → ReduceLROnPlateau  (safest, adapts automatically)
Know the total number of epochs?           → CosineAnnealingLR  (smooth, often best results)
Want the simplest schedule?                → StepLR             (predictable, easy to reason about)

ReduceLROnPlateau:   scheduler.step(val_metric)   ← pass the metric value
CosineAnnealingLR:   scheduler.step()             ← no argument needed
StepLR:              scheduler.step()             ← no argument needed
```

### Transfer Learning Decision Guide

```
Small dataset + domain similar to ImageNet?   → Feature Extraction  (freeze backbone, train head only)
Medium dataset, any domain?                   → Fine-Tuning         (train head first, then unfreeze top layers)
Large dataset + very different domain?        → Full Retraining     (all layers trainable from start)

Always preprocess with ImageNet stats for torchvision pretrained models:
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```
