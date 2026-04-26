"""Pretrained Model Pipeline with PyTorch Lightning

Purpose: A comprehensive, reusable prototype for:
- Loading a pretrained model from torchvision (ResNet50 by default)
- Three transfer learning strategies: Feature Extraction, Fine-Tuning, Full Retraining
- Full training loop managed by PyTorch Lightning
- Built-in logging, checkpointing, early stopping, LR monitoring

Sections:
    0. Imports
    1. Configuration & Reproducibility
    2. Data Pipeline — LightningDataModule
    3. Pretrained Model Builder
    4. Transfer Learning Strategies
    5. LightningModule
    6. Callbacks
    7. Trainer & Training
    8. Test Set Evaluation
    9. Plot Training Curves from CSV Log
   10. Per-Class Accuracy
   11. Inference on a Single Image
   12. Strategy Comparison (commented out)

Requirements:
    pip install lightning torchmetrics torchvision pandas
"""

# ── Non-interactive backend (must be set before any other matplotlib import) ──
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# 0. Imports
# =============================================================================
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
from torchvision import datasets
from torchvision.utils import make_grid

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

import torchmetrics

print(f'PyTorch:     {torch.__version__}')
print(f'Lightning:   {L.__version__}')
print(f'Torchvision: {torchvision.__version__}')


# =============================================================================
# 1. Configuration & Reproducibility
# =============================================================================

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
L.seed_everything(SEED, workers=True)   # Sets PyTorch, NumPy, random, and worker seeds

# ── Device (Lightning handles this automatically, but useful for manual ops) ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    # Dataset
    'dataset'         : 'CIFAR10',    # Change to match your dataset
    'num_classes'     : 10,
    'data_dir'        : './data',
    'val_split'       : 0.15,

    # Model
    'backbone'        : 'resnet50',   # resnet18, resnet50, mobilenet_v3_small, efficientnet_b0
    'pretrained'      : True,

    # Transfer learning strategy
    # Options: 'feature_extraction' | 'fine_tuning' | 'full_retraining'
    'strategy'        : 'fine_tuning',

    # Training
    'batch_size'      : 32,
    'num_epochs'      : 20,
    'head_lr'         : 1e-3,         # LR for the new classifier head
    'backbone_lr'     : 1e-5,         # LR for backbone layers (fine-tuning only)
    'weight_decay'    : 1e-4,
    'label_smoothing' : 0.1,
    'num_workers'     : 2,

    # Fine-tuning stage 2
    'unfreeze_epoch'  : 5,            # Epoch at which to unfreeze backbone layers

    # Paths
    'log_dir'         : './lightning_logs',
    'ckpt_dir'        : './checkpoints_lightning',
}

os.makedirs(CONFIG['log_dir'],  exist_ok=True)
os.makedirs(CONFIG['ckpt_dir'], exist_ok=True)
print('Config loaded.')


# =============================================================================
# 2. Data Pipeline — LightningDataModule
# =============================================================================

# ImageNet statistics — required for all torchvision pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Helper: apply a transform to a Subset that has no transform."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


class CIFAR10DataModule(L.LightningDataModule):
    """DataModule for CIFAR-10 with ImageNet-compatible preprocessing.

    Resize to 224×224 so the pretrained ResNet/MobileNet features are valid.
    For grayscale datasets, use Grayscale(num_output_channels=3) before ToTensor.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.data_dir    = cfg['data_dir']
        self.batch_size  = cfg['batch_size']
        self.num_workers = cfg['num_workers']
        self.val_split   = cfg['val_split']
        # save_hyperparameters not used here — cfg is a plain dict, not keyword args

    def prepare_data(self):
        """Download data — called once on the main process (not replicated)."""
        datasets.CIFAR10(self.data_dir, train=True,  download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """Split and assign datasets — called on every GPU."""
        # Training transform — data augmentation + ImageNet normalization
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),                     # Random 224×224 crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Validation / test transform — deterministic resize + center crop
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        if stage in ('fit', None):
            full = datasets.CIFAR10(self.data_dir, train=True, transform=None)
            n_val   = int(len(full) * self.val_split)
            n_train = len(full) - n_val
            train_sub, val_sub = random_split(
                full, [n_train, n_val],
                generator=torch.Generator().manual_seed(SEED)
            )
            # Wrap subsets with their respective transforms
            self.train_dataset = _SubsetWithTransform(train_sub, train_transform)
            self.val_dataset   = _SubsetWithTransform(val_sub,   val_transform)
            self.class_names   = full.classes

        if stage in ('test', 'predict', None):
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=val_transform
            )
            if not hasattr(self, 'class_names'):
                self.class_names = self.test_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for visualization."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1)


# =============================================================================
# 3. Pretrained Model Builder
# =============================================================================

def build_pretrained_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Load a pretrained backbone and replace its classifier head.

    Supports: resnet18, resnet50, mobilenet_v3_small, efficientnet_b0.
    Add more backbones by extending the elif chain.
    """
    # ── ResNet family ────────────────────────────────────────────────────────
    if backbone == 'resnet18':
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.resnet18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)

    elif backbone == 'resnet50':
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)

    # ── MobileNet ────────────────────────────────────────────────────────────
    elif backbone == 'mobilenet_v3_small':
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.mobilenet_v3_small(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)

    # ── EfficientNet ─────────────────────────────────────────────────────────
    elif backbone == 'efficientnet_b0':
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)

    else:
        raise ValueError(f'Unsupported backbone: {backbone}')

    return model


# =============================================================================
# 4. Transfer Learning Strategies
# =============================================================================

def apply_strategy_feature_extraction(model: nn.Module) -> None:
    """Strategy 1: Freeze entire backbone. Only the new head is trained."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the new head
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True


def apply_strategy_fine_tuning(model: nn.Module) -> None:
    """Strategy 2: Start with frozen backbone (unfreezes top layers later).

    Stage 1 (epochs 0..unfreeze_epoch-1): head only — applied here.
    Stage 2 (epochs unfreeze_epoch+):    full model — triggered in on_train_epoch_start.
    """
    apply_strategy_feature_extraction(model)   # Stage 1: head only


def apply_strategy_full_retraining(model: nn.Module) -> None:
    """Strategy 3: All layers are trainable from the start."""
    for param in model.parameters():
        param.requires_grad = True


STRATEGIES = {
    'feature_extraction': apply_strategy_feature_extraction,
    'fine_tuning'       : apply_strategy_fine_tuning,
    'full_retraining'   : apply_strategy_full_retraining,
}
print('Strategy functions defined:', list(STRATEGIES.keys()))


# =============================================================================
# 5. LightningModule
# =============================================================================

class TransferLearningModule(L.LightningModule):
    """LightningModule for pretrained model fine-tuning.

    Supports three transfer learning strategies:
    - 'feature_extraction': frozen backbone, train head only
    - 'fine_tuning':        frozen at start, unfreeze top layers at epoch N
    - 'full_retraining':    all layers trained from epoch 1
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # Store config manually — save_hyperparameters works with keyword args, not dicts
        self.cfg = cfg

        # ── Build model ───────────────────────────────────────────────────────
        self.model = build_pretrained_model(
            cfg['backbone'], cfg['num_classes'], cfg['pretrained']
        )

        # ── Apply transfer learning strategy ──────────────────────────────────
        STRATEGIES[cfg['strategy']](self.model)
        self._log_trainable_params()

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get('label_smoothing', 0.0)
        )

        # ── Metrics (torchmetrics automatically handles device placement) ─────
        n = cfg['num_classes']
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n, average='micro')
        self.val_acc   = torchmetrics.Accuracy(task='multiclass', num_classes=n, average='micro')
        self.test_acc  = torchmetrics.Accuracy(task='multiclass', num_classes=n, average='micro')
        self.val_f1    = torchmetrics.F1Score(task='multiclass',  num_classes=n, average='macro')
        self.test_f1   = torchmetrics.F1Score(task='multiclass',  num_classes=n, average='macro')
        self.val_prec  = torchmetrics.Precision(task='multiclass', num_classes=n, average='macro')
        self.val_rec   = torchmetrics.Recall(task='multiclass',   num_classes=n, average='macro')

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    # ── Training step ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.train_acc.update(preds, labels)

        # self.log() sends values to logger + progress bar
        # on_epoch=True: average over the epoch, on_step=False: don't log each step
        self.log('train/loss', loss,            on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc',  self.train_acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_prec.update(preds, labels)
        self.val_rec.update(preds, labels)

        self.log('val/loss', loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',  self.val_acc,   on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1',   self.val_f1,    on_step=False, on_epoch=True)
        self.log('val/prec', self.val_prec,  on_step=False, on_epoch=True)
        self.log('val/rec',  self.val_rec,   on_step=False, on_epoch=True)

    # ── Test step ─────────────────────────────────────────────────────────────
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.test_acc.update(preds, labels)
        self.test_f1.update(preds, labels)

        self.log('test/loss', loss,          on_step=False, on_epoch=True)
        self.log('test/acc',  self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1',   self.test_f1,  on_step=False, on_epoch=True)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    def configure_optimizers(self):
        """Separate LR for head vs backbone (for fine-tuning strategy)."""

        # Identify head vs backbone parameters
        head_params     = []
        backbone_params = []

        # Get head module
        head = getattr(self.model, 'fc', None) or getattr(self.model, 'classifier', None)
        head_param_ids = {id(p) for p in head.parameters()} if head else set()

        for p in self.model.parameters():
            if id(p) in head_param_ids:
                head_params.append(p)
            else:
                backbone_params.append(p)

        # Two parameter groups: backbone uses lower LR
        param_groups = [
            {'params': head_params,     'lr': self.cfg['head_lr']},
            {'params': backbone_params, 'lr': self.cfg['backbone_lr']},
        ]

        optimizer = optim.Adam(
            param_groups,
            weight_decay=self.cfg['weight_decay'],
        )

        # CosineAnnealing: smooth decay across all epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg['num_epochs'],
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor'  : 'val/acc',   # Only needed for ReduceLROnPlateau
                'interval' : 'epoch',
                'frequency': 1,
            },
        }

    # ── Fine-tuning: unfreeze backbone at epoch N ─────────────────────────────
    def on_train_epoch_start(self):
        """For 'fine_tuning' strategy: unfreeze backbone layers at unfreeze_epoch."""
        if self.cfg['strategy'] != 'fine_tuning':
            return

        if self.current_epoch == self.cfg['unfreeze_epoch']:
            print(f'\nEpoch {self.current_epoch}: Unfreezing backbone layers...')
            for param in self.model.parameters():
                param.requires_grad = True
            self._log_trainable_params()

    def _log_trainable_params(self):
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Parameters — Total: {total:,} | Trainable: {trainable:,} '
              f'({trainable/total:.1%})')


print('TransferLearningModule defined.')


# =============================================================================
# 6. Callbacks
# =============================================================================

# ── Checkpoint callback — saves best model based on val/acc ──────────────────
checkpoint_callback = ModelCheckpoint(
    dirpath=CONFIG['ckpt_dir'],
    filename='best-{epoch:02d}-{val/acc:.4f}',
    monitor='val/acc',
    mode='max',                  # 'max' because higher accuracy is better
    save_top_k=1,                # Keep only the best checkpoint
    save_last=True,              # Also save last epoch
    auto_insert_metric_name=False,
)

# ── Early stopping — stops if val/acc doesn't improve for patience epochs ─────
early_stopping = EarlyStopping(
    monitor='val/acc',
    mode='max',
    patience=6,
    verbose=True,
    min_delta=0.001,             # Minimum improvement to count
)

# ── LR monitor — logs current LR to CSV/TensorBoard ──────────────────────────
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# ── Logger — saves metrics to CSV ────────────────────────────────────────────
logger = CSVLogger(save_dir=CONFIG['log_dir'], name=CONFIG['backbone'])

callbacks = [checkpoint_callback, early_stopping, lr_monitor]
# Add RichProgressBar for nicer display: callbacks.append(RichProgressBar())

print('Callbacks configured.')


# =============================================================================
# 7. Trainer & Training
# =============================================================================

# Guard against multiprocessing issues on some platforms when using DataLoader workers
if __name__ == '__main__':
    import pandas as pd

    # ── Instantiate DataModule ────────────────────────────────────────────────
    dm = CIFAR10DataModule(CONFIG)
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')

    batch = next(iter(dm.train_dataloader()))
    print(f'Train batch — images: {batch[0].shape} | labels: {batch[1].shape}')
    print(f'Classes: {dm.class_names}')

    # ── Visualize a training batch ────────────────────────────────────────────
    images, labels = batch
    grid = make_grid(denormalize(images[:16]), nrow=8, padding=2)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title('Training batch (224×224, augmented, ImageNet-normalized)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_batch_lightning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: sample_batch_lightning.png')

    # ── Verify builder ────────────────────────────────────────────────────────
    _test_model = build_pretrained_model(CONFIG['backbone'], CONFIG['num_classes'], CONFIG['pretrained'])
    total_params     = sum(p.numel() for p in _test_model.parameters())
    trainable_params = sum(p.numel() for p in _test_model.parameters() if p.requires_grad)
    print(f'Backbone: {CONFIG["backbone"]}')
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    del _test_model

    # ── Instantiate Module ────────────────────────────────────────────────────
    lit_model = TransferLearningModule(CONFIG)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=CONFIG['num_epochs'],
        accelerator='auto',              # 'auto' picks GPU if available, else CPU
        devices='auto',                  # 'auto' picks all available GPUs
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,            # Log every N optimizer steps
        deterministic=True,              # Ensures reproducibility (may be slower)
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
                                         # Mixed precision on GPU; falls back to float32 on CPU
        gradient_clip_val=1.0,           # Clip gradients to prevent exploding gradients
        enable_progress_bar=True,
        enable_model_summary=True,       # Print model summary before training
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(lit_model, datamodule=dm)

    print(f'\nBest model checkpoint: {checkpoint_callback.best_model_path}')
    print(f'Best val/acc:          {checkpoint_callback.best_model_score:.4f}')

    # =========================================================================
    # 8. Test Set Evaluation
    # =========================================================================

    # ── Load the best checkpoint and evaluate on the test set ─────────────────
    print('Loading best checkpoint for test evaluation...')
    test_results = trainer.test(
        lit_model,
        datamodule=dm,
        ckpt_path='best',       # 'best' loads checkpoint_callback.best_model_path automatically
    )

    print('\nTest Results:')
    for k, v in test_results[0].items():
        print(f'  {k:<20}: {v:.4f}')

    # =========================================================================
    # 9. Plot Training Curves from CSV Log
    # =========================================================================

    log_path = Path(CONFIG['log_dir']) / CONFIG['backbone']
    csv_files = sorted(log_path.glob('version_*/metrics.csv'))
    if not csv_files:
        print('No CSV log found yet.')
    else:
        df = pd.read_csv(csv_files[-1])
        print(f'Loaded metrics from: {csv_files[-1]}')
        print(df.head())

        # Separate epoch-level metrics
        df_epoch = df.dropna(subset=['epoch']).groupby('epoch').last().reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Loss
        if 'train/loss_epoch' in df_epoch.columns and 'val/loss' in df_epoch.columns:
            axes[0].plot(df_epoch['epoch'], df_epoch['train/loss_epoch'], label='Train Loss', color='steelblue')
            axes[0].plot(df_epoch['epoch'], df_epoch['val/loss'],         label='Val Loss',   color='tomato')
            axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Accuracy
        if 'train/acc_epoch' in df_epoch.columns and 'val/acc' in df_epoch.columns:
            axes[1].plot(df_epoch['epoch'], df_epoch['train/acc_epoch'] * 100, label='Train Acc', color='steelblue')
            axes[1].plot(df_epoch['epoch'], df_epoch['val/acc'] * 100,         label='Val Acc',   color='tomato')
            axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # Learning rate
        lr_col = [c for c in df_epoch.columns if 'lr' in c.lower()]
        if lr_col:
            axes[2].plot(df_epoch['epoch'], df_epoch[lr_col[0]], color='green')
            axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
            axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'{CONFIG["backbone"]} — {CONFIG["strategy"]}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lightning_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved: lightning_training_curves.png')

    # =========================================================================
    # 10. Per-Class Accuracy
    # =========================================================================

    # ── Load best model from checkpoint ──────────────────────────────────────
    best_module = TransferLearningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=CONFIG,
    )
    best_module.eval()
    best_module.to(device)

    # ── Collect predictions ──────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dm.test_dataloader():
            inputs  = inputs.to(device)
            logits  = best_module(inputs)
            preds   = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds  = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    CLASS_NAMES = dm.class_names

    # Per-class accuracy
    print(f'\nPer-class accuracy (Test set):')
    print(f'{"Class":<15} {"Accuracy":>10}')
    print('-' * 28)
    class_accs = []
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        acc  = (all_preds[mask] == all_labels[mask]).float().mean().item()
        class_accs.append(acc)
        print(f'{name:<15} {acc:>10.2%}')

    # Bar chart
    overall_acc = (all_preds == all_labels).float().mean().item()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(CLASS_NAMES, [a * 100 for a in class_accs], color='steelblue', edgecolor='white')
    ax.axhline(overall_acc * 100, color='tomato', linestyle='--', label=f'Overall: {overall_acc:.1%}')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Per-Class Accuracy — {CONFIG["backbone"]} ({CONFIG["strategy"]})')
    ax.set_ylim(0, 100)
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('per_class_accuracy_lightning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: per_class_accuracy_lightning.png')

    # =========================================================================
    # 11. Inference on a Single Image
    # =========================================================================

    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def predict_image(model, image, class_names, device, top_k=5):
        """Inference on a single PIL image or file path.

        Args:
            model:       nn.Module (set to eval mode before calling)
            image:       PIL.Image or str file path
            class_names: list of class name strings
            device:      torch.device
            top_k:       number of top predictions to return

        Returns:
            list of (class_name, probability) sorted by probability descending
        """
        from PIL import Image as PILImage
        if isinstance(image, str):
            image = PILImage.open(image).convert('RGB')

        model.eval()
        with torch.no_grad():
            tensor = inference_transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = probs.topk(top_k)
        return [(class_names[i.item()], p.item()) for i, p in zip(top_indices, top_probs)]

    # Demo: first image from test dataset
    sample_tensor, sample_label = dm.test_dataset[0]
    # Denormalize for display
    sample_display = denormalize(sample_tensor)

    # Get raw PIL image from the base (untransformed) dataset for inference
    raw_pil_image, _ = dm.test_dataset.dataset[0]   # raw PIL Image before any transform
    predictions = predict_image(best_module.model, raw_pil_image, CLASS_NAMES, device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(sample_display.permute(1, 2, 0).numpy())
    ax1.set_title(f'Ground truth: {CLASS_NAMES[sample_label]}')
    ax1.axis('off')

    cls_names = [p[0] for p in predictions]
    probs     = [p[1] * 100 for p in predictions]
    colors    = ['limegreen' if c == CLASS_NAMES[sample_label] else 'steelblue' for c in cls_names]
    ax2.barh(cls_names[::-1], probs[::-1], color=colors[::-1])
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top-5 Predictions')
    ax2.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig('inference_demo_lightning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: inference_demo_lightning.png')

    print('\nTop-5 predictions:')
    for cls, prob in predictions:
        marker = ' <- GT' if cls == CLASS_NAMES[sample_label] else ''
        print(f'  {cls:<15}: {prob:.2%}{marker}')

    # =========================================================================
    # 12. Strategy Comparison (Optional)
    # =========================================================================

    # Uncomment to run a comparison across all three strategies.
    # This trains the model 3× — only run if you have time/GPU.

    # results = {}
    # for strategy in ['feature_extraction', 'fine_tuning', 'full_retraining']:
    #     cfg = {**CONFIG, 'strategy': strategy, 'num_epochs': 10}
    #     dm_s    = CIFAR10DataModule(cfg)
    #     model_s = TransferLearningModule(cfg)
    #     trainer_s = L.Trainer(
    #         max_epochs=cfg['num_epochs'],
    #         accelerator='auto', devices='auto',
    #         enable_progress_bar=True, enable_model_summary=False,
    #         logger=False, callbacks=[EarlyStopping('val/acc', mode='max', patience=4)],
    #     )
    #     trainer_s.fit(model_s, datamodule=dm_s)
    #     test_res = trainer_s.test(model_s, datamodule=dm_s, verbose=False)
    #     results[strategy] = test_res[0].get('test/acc', 0)
    #     print(f'{strategy}: test_acc={results[strategy]:.4f}')
    #
    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax.bar(results.keys(), [v*100 for v in results.values()], color=['steelblue','tomato','limegreen'])
    # ax.set_ylabel('Test Accuracy (%)')
    # ax.set_title('Strategy Comparison')
    # plt.tight_layout()
    # plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    # plt.close()

    print('\nDone.')
