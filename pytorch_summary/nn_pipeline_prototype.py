"""
Neural Network Pipeline Prototype
==================================
Purpose : Comprehensive, reusable prototype for image classification from scratch.
          No pretrained models. No Lightning module.
Dataset : CIFAR-10 (auto-downloaded). Swap for your own via CustomImageDataset
          or datasets.ImageFolder — only the data-loading section needs changing.
Sections:
  0. Imports
  1. Configuration & Reproducibility
  2. Data Pipeline (Transforms, Dataset, DataLoaders)
  3. Model Architecture (CNN)
  4. Loss, Optimizer, Scheduler
  5. Training Loop
  6. Plot Training Curves
  7. Test Set Evaluation & Metrics
  8. Save & Load Model
  9. Inference on a Single Image

Run:
  python nn_pipeline_prototype.py

Requirements:
  pip install torch torchvision matplotlib pillow numpy
  pip install torchmetrics   # optional — for Precision/Recall/F1
"""

# ==============================================================================
# 0. Imports
# ==============================================================================
import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # Use non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid

from PIL import Image

# torchmetrics is optional — gracefully degrade if not installed
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print('torchmetrics not installed — using manual accuracy only')
    print('Install with: pip install torchmetrics')

print(f'PyTorch version:      {torch.__version__}')
print(f'Torchvision version:  {torchvision.__version__}')


# ==============================================================================
# 1. Configuration & Reproducibility
# ==============================================================================

SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Auto-detect GPU; fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Centralised config dict — change values here, not scattered through the code
CONFIG = {
    'num_classes'    : 10,
    'image_size'     : 32,       # Resize all images to this (H = W)
    'in_channels'    : 3,        # 3 for RGB, 1 for grayscale
    'batch_size'     : 64,
    'num_epochs'     : 20,
    'learning_rate'  : 1e-3,
    'weight_decay'   : 1e-4,     # L2 regularisation coefficient
    'dropout_rate'   : 0.5,
    'num_workers'    : 2,        # Parallel DataLoader workers
    'data_dir'       : './data',
    'checkpoint_dir' : './checkpoints',
    'val_split'      : 0.15,     # Fraction of training data used for validation
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
print('\nConfig:', CONFIG)


# ==============================================================================
# 2. Data Pipeline
# ==============================================================================

# ── 2a. Dataset statistics ────────────────────────────────────────────────────
# Precomputed per-channel mean and std for CIFAR-10 training set.
# Replace with your own dataset's statistics if not using CIFAR-10.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ── 2b. Transforms ────────────────────────────────────────────────────────────
# Transform order rule:
#   1. Geometric  (Resize, Crop, Flip, Rotate)  — on PIL image
#   2. Colour     (ColorJitter)                 — on PIL image
#   3. ToTensor() — converts PIL → tensor, scales [0,255] → [0,1], HWC → CHW
#   4. Normalize  — zero-centres each channel using mean/std
# Augmentation is applied ONLY to training data.

train_transform = transforms.Compose([
    transforms.RandomCrop(CONFIG['image_size'], padding=4),  # Pad then crop randomly
    transforms.RandomHorizontalFlip(p=0.5),                  # Mirror 50% of images
    transforms.RandomRotation(degrees=10),                    # ±10° rotation
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.1, hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Validation/test: deterministic — no random transforms
val_transform = transforms.Compose([
    transforms.Resize(CONFIG['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

print('\nTrain transform:', train_transform)
print('\nVal transform:  ', val_transform)


# ── 2c. Dataset classes ───────────────────────────────────────────────────────

class SubsetWithTransform(Dataset):
    """Apply a specific transform to a Subset returned by random_split().

    random_split() returns Subset objects that share the parent dataset's
    transform. This wrapper lets each split (train/val) use its own transform.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomImageDataset(Dataset):
    """Template for loading images from a folder tree.

    Expected folder structure:
        root/
            class_a/image1.jpg
            class_b/image1.jpg
            ...

    To use instead of CIFAR-10:
        dataset = CustomImageDataset('path/to/root', transform=train_transform)
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()   # List of (path, label_index)

    def _find_classes(self):
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for path in cls_dir.iterdir():
                if path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    samples.append((path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')   # Always convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label


# ── 2d. Load data ─────────────────────────────────────────────────────────────
# Using CIFAR-10 as the demonstration dataset.
# To use your own data, replace these two lines:
#   full_dataset     = CustomImageDataset('path/to/train', transform=None)
#   test_dataset_raw = CustomImageDataset('path/to/test',  transform=None)

full_dataset = datasets.CIFAR10(
    root=CONFIG['data_dir'],
    train=True,
    download=True,
    transform=None    # Transform assigned per split below
)
test_dataset_raw = datasets.CIFAR10(
    root=CONFIG['data_dir'],
    train=False,
    download=True,
    transform=None
)

CLASS_NAMES = full_dataset.classes
print(f'\nClasses ({len(CLASS_NAMES)}): {CLASS_NAMES}')
print(f'Train+Val samples: {len(full_dataset)}')
print(f'Test samples:      {len(test_dataset_raw)}')

# Split train → train / val
n_total = len(full_dataset)
n_val   = int(n_total * CONFIG['val_split'])
n_train = n_total - n_val

train_subset, val_subset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

# Wrap each split with its own transform
train_dataset = SubsetWithTransform(train_subset, train_transform)
val_dataset   = SubsetWithTransform(val_subset,   val_transform)
test_dataset  = SubsetWithTransform(test_dataset_raw, val_transform)

print(f'\nSplit sizes — Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

# ── 2e. DataLoaders ───────────────────────────────────────────────────────────
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,                       # Randomise batch order each epoch
    num_workers=CONFIG['num_workers'],
    pin_memory=(device.type == 'cuda'), # Faster CPU→GPU transfer
    drop_last=True,                     # Avoids single-sample batches (bad for BatchNorm)
    persistent_workers=(CONFIG['num_workers'] > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'] * 2,  # Larger batch OK — no gradient storage needed
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=(device.type == 'cuda'),
    persistent_workers=(CONFIG['num_workers'] > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'] * 2,
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=(device.type == 'cuda'),
)

print(f'Batches per epoch — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')


# ── 2f. Visualise a batch ─────────────────────────────────────────────────────

def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Reverse normalisation for display purposes."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# Peek at one training batch
images, labels = next(iter(train_loader))
print(f'\nBatch shapes — Images: {images.shape} | Labels: {labels.shape}')

fig, ax = plt.subplots(figsize=(12, 6))
grid = make_grid(denormalize(images[:32]), nrow=8, padding=2)
ax.imshow(grid.permute(1, 2, 0).numpy())
ax.set_title('Training batch sample (after augmentation)')
ax.axis('off')
plt.tight_layout()
plt.savefig('sample_batch.png', dpi=150, bbox_inches='tight')
plt.close()

print('Labels:', [CLASS_NAMES[l] for l in labels[:8].tolist()])


# ==============================================================================
# 3. Model Architecture
# ==============================================================================

class CNN(nn.Module):
    """Flexible CNN for image classification.

    Architecture: 3 conv blocks (Conv → BatchNorm → ReLU → MaxPool) + 2 FC layers.
    Channels grow:     in_channels → 32 → 64 → 128
    Spatial dims shrink:  image_size → /2 → /4 → /8
    Output: raw logits (no softmax) — CrossEntropyLoss applies log-softmax internally.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1 — in_channels → 32 channels, image_size → image_size/2
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),       # Normalise activations; acts as regulariser
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 — 32 → 64 channels, /2 → /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 — 64 → 128 channels, /4 → /8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Compute the flattened feature size dynamically using a dummy forward pass.
        # This avoids hardcoding and automatically adapts to any image_size.
        dummy     = torch.zeros(1, in_channels, CONFIG['image_size'], CONFIG['image_size'])
        flat_size = self.features(dummy).view(1, -1).shape[1]

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),         # Drop 50% of neurons — prevents memorisation
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),   # Raw logits — no softmax
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming (He) initialisation — recommended for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Instantiate and move to device
model = CNN(
    num_classes=CONFIG['num_classes'],
    in_channels=CONFIG['in_channels'],
    dropout=CONFIG['dropout_rate'],
).to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel: CNN')
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Sanity-check forward pass shape
test_input  = torch.randn(2, CONFIG['in_channels'], CONFIG['image_size'], CONFIG['image_size']).to(device)
test_output = model(test_input)
print(f'Forward pass — Input: {test_input.shape} → Output: {test_output.shape}')


# ==============================================================================
# 4. Loss Function, Optimizer & Scheduler
# ==============================================================================

# CrossEntropyLoss = log-softmax + NLLLoss (do NOT apply softmax before this)
# label_smoothing=0.1 softens targets: prevents overconfidence, improves calibration
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Adam: adaptive per-parameter LR, converges faster than SGD for most tasks
# weight_decay applies L2 regularisation (penalises large weights)
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
)

# ReduceLROnPlateau: reduces LR when monitored metric stops improving
# mode='max' because we monitor accuracy (higher is better)
# patience=3: wait 3 epochs with no improvement before reducing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,      # New LR = LR × 0.5
    patience=3,
    min_lr=1e-6,
    verbose=True,
)

print('\nCriterion:', criterion)
print('Optimizer:', optimizer)


# ==============================================================================
# 5. Training Loop
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (mean_loss, accuracy)."""
    model.train()    # Activates Dropout and BatchNorm batch-stats mode
    total_loss = 0.0
    correct = total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()              # 1. Clear gradients from previous batch
        outputs = model(inputs)            # 2. Forward pass
        loss    = criterion(outputs, labels)  # 3. Compute loss
        loss.backward()                    # 4. Backpropagate
        optimizer.step()                   # 5. Update weights

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataloader. Returns (mean_loss, accuracy).
    @torch.no_grad() disables gradient computation — saves memory, speeds up eval.
    """
    model.eval()     # Deactivates Dropout; switches BatchNorm to running statistics
    total_loss = 0.0
    correct = total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


# ── Training state ─────────────────────────────────────────────────────────────
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss'  : [], 'val_acc'  : [],
    'lr'        : [],
}

best_val_acc       = 0.0
patience_counter   = 0
EARLY_STOP_PATIENCE = 8
checkpoint_path    = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')

# ── Main training loop ─────────────────────────────────────────────────────────
print(f'\nTraining for {CONFIG["num_epochs"]} epochs on {device}')
print('-' * 75)
print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>9} | '
      f'{"Val Loss":>8} | {"Val Acc":>8} | {"LR":>9}')
print('-' * 75)

start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)    # Must be called AFTER validation with the monitored metric

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    print(f'{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | '
          f'{val_loss:>8.4f} | {val_acc:>8.2%} | {current_lr:>9.2e}')

    # Save checkpoint when val accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch'              : epoch,
            'model_state_dict'   : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc'            : best_val_acc,
            'config'             : CONFIG,
        }, checkpoint_path)
        print(f'         ↑ New best val acc: {best_val_acc:.2%} — checkpoint saved')
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'\nEarly stopping at epoch {epoch} — no improvement for {EARLY_STOP_PATIENCE} epochs')
            break

elapsed = time.time() - start_time
print('-' * 75)
print(f'Training complete in {elapsed / 60:.1f} min | Best Val Acc: {best_val_acc:.2%}')


# ==============================================================================
# 6. Plot Training Curves
# ==============================================================================

epochs_ran = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss — a widening gap between train and val loss signals overfitting
axes[0].plot(epochs_ran, history['train_loss'], label='Train Loss', color='steelblue')
axes[0].plot(epochs_ran, history['val_loss'],   label='Val Loss',   color='tomato')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_ran, [a * 100 for a in history['train_acc']], label='Train Acc', color='steelblue')
axes[1].plot(epochs_ran, [a * 100 for a in history['val_acc']],   label='Val Acc',   color='tomato')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Learning rate — should decay when val accuracy plateaus
axes[2].plot(epochs_ran, history['lr'], color='green')
axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

plt.suptitle('Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved training_history.png')


# ==============================================================================
# 7. Test Set Evaluation & Metrics
# ==============================================================================

# Load the best checkpoint before evaluating — not the last epoch weights
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'\nLoaded best model from epoch {checkpoint["epoch"]} (val_acc={checkpoint["val_acc"]:.2%})')

# Collect all predictions in one pass
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

all_preds  = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

# Overall accuracy
test_acc = (all_preds == all_labels).float().mean().item()
print(f'\nTest Accuracy: {test_acc:.2%}')

# Per-class accuracy — reveals which classes the model struggles with
print('\nPer-class accuracy:')
print(f'{"Class":<15} {"Correct":>8} {"Total":>8} {"Accuracy":>10}')
print('-' * 45)
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask    = all_labels == cls_idx
    correct = (all_preds[mask] == all_labels[mask]).sum().item()
    total   = mask.sum().item()
    acc     = correct / total if total > 0 else 0.0
    print(f'{cls_name:<15} {correct:>8} {total:>8} {acc:>9.2%}')

# Optional torchmetrics: Precision, Recall, F1
if TORCHMETRICS_AVAILABLE:
    n_cls = CONFIG['num_classes']
    metrics = {
        'Accuracy (micro)' : torchmetrics.Accuracy(task='multiclass', num_classes=n_cls, average='micro'),
        'Precision (macro)': torchmetrics.Precision(task='multiclass', num_classes=n_cls, average='macro'),
        'Recall (macro)'   : torchmetrics.Recall(task='multiclass', num_classes=n_cls, average='macro'),
        'F1 (macro)'       : torchmetrics.F1Score(task='multiclass', num_classes=n_cls, average='macro'),
    }
    print('\nTorchMetrics results:')
    for name, metric in metrics.items():
        value = metric(all_preds, all_labels).item()
        print(f'  {name:<25}: {value:.4f}')

# Per-class accuracy bar chart
class_accs = []
for cls_idx in range(len(CLASS_NAMES)):
    mask = all_labels == cls_idx
    acc  = (all_preds[mask] == all_labels[mask]).float().mean().item()
    class_accs.append(acc)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(CLASS_NAMES, [a * 100 for a in class_accs], color='steelblue', edgecolor='white')
ax.axhline(test_acc * 100, color='tomato', linestyle='--', label=f'Overall: {test_acc:.1%}')
ax.set_ylabel('Accuracy (%)'); ax.set_title('Per-Class Test Accuracy')
ax.set_ylim(0, 100); ax.legend()
for bar, acc in zip(bars, class_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{acc:.0%}', ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved per_class_accuracy.png')


# ==============================================================================
# 8. Save & Load Model
# ==============================================================================

final_save_path = os.path.join(CONFIG['checkpoint_dir'], 'final_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names'     : CLASS_NAMES,
    'config'          : CONFIG,
}, final_save_path)
print(f'\nModel saved to {final_save_path}')


def load_model(path: str, device: torch.device):
    """Load a saved CNN model from checkpoint.

    Args:
        path:   Path to the .pth checkpoint file
        device: Target device (cpu or cuda)
    Returns:
        (model, class_names) tuple
    """
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt['config']
    model = CNN(
        num_classes=cfg['num_classes'],
        in_channels=cfg['in_channels'],
        dropout=cfg['dropout_rate'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('class_names', [])


loaded_model, loaded_class_names = load_model(final_save_path, device)
print(f'Model loaded. Classes: {loaded_class_names}')


# ==============================================================================
# 9. Inference on a Single Image
# ==============================================================================

def predict_single(model, image_tensor: torch.Tensor, class_names: list,
                   device: torch.device, top_k: int = 5):
    """Run inference on a single pre-processed image tensor.

    Args:
        model:        Trained CNN (in eval mode)
        image_tensor: Normalised tensor of shape [C, H, W]
        class_names:  List of class name strings
        device:       Target device
        top_k:        Number of top predictions to return
    Returns:
        List of (class_name, probability) sorted by probability descending
    """
    model.eval()
    with torch.no_grad():
        inp    = image_tensor.unsqueeze(0).to(device)   # [C,H,W] → [1,C,H,W]
        logits = model(inp)                              # [1, num_classes]
        probs  = torch.softmax(logits, dim=1)[0]         # [num_classes]

    top_probs, top_indices = probs.topk(top_k)
    return [(class_names[idx.item()], p.item()) for idx, p in zip(top_indices, top_probs)]


# Demo on first test image (already transformed tensor)
sample_image, sample_label = test_dataset[0]
predictions = predict_single(loaded_model, sample_image, CLASS_NAMES, device, top_k=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

img_display = denormalize(sample_image)
ax1.imshow(img_display.permute(1, 2, 0).numpy())
ax1.set_title(f'Ground truth: {CLASS_NAMES[sample_label]}')
ax1.axis('off')

pred_classes = [p[0] for p in predictions]
pred_probs   = [p[1] * 100 for p in predictions]
colors = ['limegreen' if p[0] == CLASS_NAMES[sample_label] else 'steelblue' for p in predictions]
ax2.barh(pred_classes[::-1], pred_probs[::-1], color=colors[::-1])
ax2.set_xlabel('Probability (%)')
ax2.set_title('Top-5 Predictions')
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('inference_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print('\nTop-5 predictions:')
for cls, prob in predictions:
    marker = ' ← GT' if cls == CLASS_NAMES[sample_label] else ''
    print(f'  {cls:<15}: {prob:.2%}{marker}')

print('\nDone. Output files: training_history.png, per_class_accuracy.png, inference_demo.png')
