"""
C3M4 Assignment — exercise solutions (pruning, dynamic quantization, fusion, QAT).

Copy each function body into the matching graded cell in C3M4_Assignment.ipynb
(between ### START CODE HERE ### and ### END CODE HERE ###).

Requires the course files in the same directory when running tests:
  helper_utils.py, unittests.py, street_classifier_weights.pt, data/
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.nn.utils import prune


# ---------------------------------------------------------------------------
# Exercise 1 — depends on helpers defined in the assignment notebook
# ---------------------------------------------------------------------------

def prune_model(model, amount=0.3, mode="l1_unstructured"):
  """Apply magnitude-based pruning to Conv2d and Linear weights (in-place)."""
  if not (0.0 <= amount <= 1.0):
    raise ValueError(f"amount must be in [0,1], got {amount}")

  for _, module in _iter_prunable_modules(model):
    if not hasattr(module, "weight"):
      continue

    if mode == "l1_unstructured":
      prune.l1_unstructured(module, name="weight", amount=amount)
    elif mode == "ln_structured":
      prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    else:
      raise ValueError("mode must be 'l1_unstructured' or 'ln_structured'")

  return model


# ---------------------------------------------------------------------------
# Exercise 2
# ---------------------------------------------------------------------------

def quantize_dynamic_linear(model):
  """Return a deep-copied model with nn.Linear layers dynamically quantized to INT8."""
  model_fp32 = copy.deepcopy(model).eval()

  has_quantized = hasattr(torch.backends, "quantized")
  has_engine = hasattr(torch.backends.quantized, "engine") if has_quantized else False
  if has_quantized and has_engine:
    try:
      torch.backends.quantized.engine = "fbgemm"
    except Exception:
      pass

  quantized = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8,
  )
  quantized.eval()
  return quantized


# ---------------------------------------------------------------------------
# Exercise 3
# ---------------------------------------------------------------------------

def fuse_model_inplace(model: nn.Module) -> nn.Module:
  """Recursively fuse Conv/BN/ReLU and Linear/ReLU patterns inside nn.Sequential blocks."""
  for _, child in model.named_children():
    fuse_model_inplace(child)

    if isinstance(child, nn.Sequential) and len(child) >= 2:
      was_training = child.training
      child.eval()
      i = 0
      while i < len(child) - 1:
        a, b = child[i], child[i + 1]
        c = child[i + 2] if i + 2 < len(child) else None

        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU):
          torch.quantization.fuse_modules(child, [str(i), str(i + 1), str(i + 2)], inplace=True)
          i += 3
          continue

        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
          torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
          i += 2
          continue

        if isinstance(a, nn.Conv2d) and isinstance(b, nn.ReLU):
          torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
          i += 2
          continue

        if isinstance(a, nn.Linear) and isinstance(b, nn.ReLU):
          torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
          i += 2
          continue

        i += 1

      if was_training:
        child.train()

  return model


# ---------------------------------------------------------------------------
# Exercise 4
# ---------------------------------------------------------------------------

def prepare_qat(model, backend="fbgemm"):
  """Return a QAT-ready deep copy with fusion, qconfig, and fake-quant observers."""
  qat = copy.deepcopy(model).train()

  if hasattr(torch.backends, "quantized") and hasattr(torch.backends.quantized, "engine"):
    try:
      torch.backends.quantized.engine = backend
    except Exception:
      pass

  fuse_model_inplace(qat)
  qat.qconfig = torch.quantization.get_default_qat_qconfig(backend)
  qat.train()
  torch.quantization.prepare_qat(qat, inplace=True)
  return qat


# ---------------------------------------------------------------------------
# Notebook paste blocks (minimal — only the START/END regions)
# ---------------------------------------------------------------------------

EXERCISE_1_PASTE = '''
    if not (0.0 <= amount <= 1.0):
        raise ValueError(f"amount must be in [0,1], got {amount}")

    for _, module in _iter_prunable_modules(model):
        if not hasattr(module, "weight"):
            continue

        if mode == "l1_unstructured":
            prune.l1_unstructured(module, name="weight", amount=amount)
        elif mode == "ln_structured":
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        else:
            raise ValueError("mode must be 'l1_unstructured' or 'ln_structured'")
'''

EXERCISE_2_PASTE = '''
    model_fp32 = copy.deepcopy(model).eval()

    has_quantized = hasattr(torch.backends, "quantized")
    has_engine = hasattr(torch.backends.quantized, "engine") if has_quantized else False
    if has_quantized and has_engine:
        try:
            torch.backends.quantized.engine = "fbgemm"
        except Exception:
            pass

    quantized = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()

    return quantized
'''

EXERCISE_3_PASTE = '''
    for _, child in model.named_children():
        fuse_model_inplace(child)

        if isinstance(child, nn.Sequential) and len(child) >= 2:
            was_training = child.training
            child.eval()
            i = 0
            while i < len(child) - 1:
                a, b = child[i], child[i + 1]
                c = child[i + 2] if i + 2 < len(child) else None

                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU):
                    torch.quantization.fuse_modules(child, [str(i), str(i + 1), str(i + 2)], inplace=True)
                    i += 3
                    continue

                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                    continue

                if isinstance(a, nn.Conv2d) and isinstance(b, nn.ReLU):
                    torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                    continue

                if isinstance(a, nn.Linear) and isinstance(b, nn.ReLU):
                    torch.quantization.fuse_modules(child, [str(i), str(i + 1)], inplace=True)
                    i += 2
                    continue

                i += 1

            if was_training:
                child.train()

    return model
'''

EXERCISE_4_PASTE = '''
    qat = copy.deepcopy(model).train()

    if hasattr(torch.backends, "quantized") and hasattr(torch.backends.quantized, "engine"):
        try:
            torch.backends.quantized.engine = backend
        except Exception:
            pass

    fuse_model_inplace(qat)
    qat.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    qat.train()
    torch.quantization.prepare_qat(qat, inplace=True)
'''
