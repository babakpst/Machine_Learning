#!/usr/bin/env python3
"""Visual comparison of BatchNorm vs LayerNorm normalization axes."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT_DIR = Path(__file__).parent / "layernorm_batchnorm_figures"
OUT_DIR.mkdir(exist_ok=True)

# Colors
C_BATCH = "#4C78A8"      # blue - batch axis
C_SPATIAL = "#72B7B2"    # teal - spatial
C_CHANNEL = "#F58518"    # orange - channel / feature
C_NORM = "#E45756"       # red - normalized slice highlight
C_BG = "#F7F7F7"
C_TEXT = "#2F2F2F"


def add_tensor_block(ax, origin, shape, labels, highlight=None, title=None):
    """Draw a 3D-ish tensor block with axis labels."""
    ox, oy = origin
    n, c, h, w = shape
    scale = 0.18

    # Front face (N x H)
    fw, fh = w * scale, h * scale
    front = FancyBboxPatch(
        (ox, oy), fw, fh,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="white", edgecolor="#333", linewidth=1.5, zorder=3,
    )
    ax.add_patch(front)

    # Depth for C
    dx, dy = c * scale * 0.35, c * scale * 0.35
    top = plt.Polygon(
        [(ox, oy + fh), (ox + dx, oy + fh + dy), (ox + fw + dx, oy + fh + dy), (ox + fw, oy + fh)],
        closed=True, facecolor="#DDD", edgecolor="#333", linewidth=1.2, zorder=2,
    )
    side = plt.Polygon(
        [(ox + fw, oy), (ox + fw + dx, oy + dy), (ox + fw + dx, oy + fh + dy), (ox + fw, oy + fh)],
        closed=True, facecolor="#BBB", edgecolor="#333", linewidth=1.2, zorder=2,
    )
    ax.add_patch(top)
    ax.add_patch(side)

    if highlight == "batch_spatial":
        # Highlight one channel slice - stats over N,H,W
        for i in range(min(n, 4)):
            for j in range(min(h, 3)):
                rx = ox + 0.05 + j * (fw - 0.1) / max(h - 1, 1)
                ry = oy + 0.05 + i * (fh - 0.1) / max(n - 1, 1)
                ax.add_patch(Rectangle(
                    (rx, ry), (fw - 0.1) / h * 0.9, (fh - 0.1) / n * 0.9,
                    facecolor=C_NORM, alpha=0.35, edgecolor=C_NORM, linewidth=0.8, zorder=4,
                ))
        ax.text(ox + fw / 2, oy - 0.35, "One channel c:\nμ, σ² over N × H × W",
                ha="center", va="top", fontsize=10, color=C_NORM, fontweight="bold")

    if title:
        ax.text(ox + fw / 2 + dx / 2, oy + fh + dy + 0.25, title,
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=C_TEXT)

    # Axis arrows / labels
    ax.annotate("", xy=(ox + fw + 0.15, oy - 0.05), xytext=(ox - 0.05, oy - 0.05),
                arrowprops=dict(arrowstyle="<->", color=C_SPATIAL, lw=2))
    ax.text(ox + fw / 2, oy - 0.22, labels.get("w", "W"), ha="center", fontsize=10, color=C_SPATIAL)

    ax.annotate("", xy=(ox - 0.15, oy + fh + 0.05), xytext=(ox - 0.15, oy - 0.05),
                arrowprops=dict(arrowstyle="<->", color=C_SPATIAL, lw=2))
    ax.text(ox - 0.35, oy + fh / 2, labels.get("h", "H"), ha="center", va="center",
            fontsize=10, color=C_SPATIAL, rotation=90)

    ax.annotate("", xy=(ox + fw + dx + 0.2, oy + dy), xytext=(ox + fw + 0.05, oy),
                arrowprops=dict(arrowstyle="->", color=C_CHANNEL, lw=2))
    ax.text(ox + fw + dx + 0.35, oy + dy / 2, labels.get("c", "C"), ha="left", fontsize=10, color=C_CHANNEL)

    ax.annotate("", xy=(ox - 0.55, oy + fh + 0.55), xytext=(ox - 0.55, oy + 0.1),
                arrowprops=dict(arrowstyle="<->", color=C_BATCH, lw=2))
    ax.text(ox - 0.75, oy + fh / 2 + 0.2, labels.get("n", "N"), ha="center", va="center",
            fontsize=10, color=C_BATCH, rotation=90)


def fig_batchnorm_concept():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1.5, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(4, 4.6, "BatchNorm2d — normalize per channel over batch + spatial dims",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)

    add_tensor_block(
        ax, origin=(1.5, 1.2), shape=(4, 64, 3, 3),
        labels={"n": "N (batch)", "c": "C (channels)", "h": "H", "w": "W"},
        highlight="batch_spatial",
        title="Input: (N, C, H, W)",
    )

    # Mini grid showing one channel across batch
    gx, gy = 5.2, 2.0
    ax.text(gx + 1.2, 3.8, "Channel c=0 across batch", ha="center", fontsize=11, fontweight="bold")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (4, 3, 3))
    for b in range(4):
        for i in range(3):
            for j in range(3):
                val = data[b, i, j]
                color = plt.cm.RdBu_r((val + 2) / 4)
                ax.add_patch(Rectangle(
                    (gx + j * 0.35 + b * 1.3, gy + i * 0.35), 0.32, 0.32,
                    facecolor=color, edgecolor="#555", linewidth=0.5,
                ))
        ax.text(gx + 0.5 + b * 1.3, gy - 0.25, f"N={b}", ha="center", fontsize=9)

    ax.add_patch(FancyBboxPatch(
        (gx - 0.1, gy - 0.35), 5.3, 1.45,
        boxstyle="round,pad=0.05", facecolor=C_NORM, alpha=0.12, edgecolor=C_NORM, linewidth=2,
    ))
    ax.text(gx + 2.5, gy + 1.15, "μ_c, σ²_c computed here → same γ_c, β_c for channel c",
            ha="center", fontsize=10, color=C_NORM, fontweight="bold")

    legend = [
        mpatches.Patch(color=C_BATCH, label="Batch axis (N)"),
        mpatches.Patch(color=C_SPATIAL, label="Spatial axes (H, W)"),
        mpatches.Patch(color=C_CHANNEL, label="Channel axis (C) — one γ, β per channel"),
        mpatches.Patch(color=C_NORM, alpha=0.4, label="Dimensions averaged for stats"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=2, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    path = OUT_DIR / "01_batchnorm_concept.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_layernorm_concept():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 5.1, "LayerNorm — normalize per sample over feature dim D",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)

    # Draw batch of sequences as grid: rows = N, cols = L tokens, each token is D-wide bar
    N, L, D = 4, 6, 8
    token_w, token_h = 0.55, 0.35
    start_x, start_y = 1.0, 3.2

    ax.text(start_x + L * token_w / 2, 4.5, "Input: (N, L, D)", ha="center", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(1)
    for n in range(N):
        y = start_y - n * 0.55
        ax.text(start_x - 0.45, y + token_h / 2, f"N={n}", ha="right", va="center", fontsize=9, color=C_BATCH)
        for l in range(L):
            x = start_x + l * token_w
            # D features as vertical stripes inside token
            for d in range(D):
                val = rng.normal(0, 1)
                color = plt.cm.RdBu_r((val + 2) / 4)
                ax.add_patch(Rectangle(
                    (x + d * (token_w / D), y), token_w / D - 0.01, token_h,
                    facecolor=color, edgecolor="none",
                ))
            ax.add_patch(Rectangle(
                (x, y), token_w - 0.02, token_h,
                facecolor="none", edgecolor="#444", linewidth=0.8,
            ))
            if n == 0:
                ax.text(x + token_w / 2, y + token_h + 0.12, f"L={l}", ha="center", fontsize=7)

    # Highlight one token's D features
    hx, hy = start_x + 2 * token_w, start_y - 1 * 0.55
    ax.add_patch(FancyBboxPatch(
        (hx - 0.05, hy - 0.08), token_w + 0.1, token_h + 0.16,
        boxstyle="round,pad=0.02", facecolor=C_NORM, alpha=0.25, edgecolor=C_NORM, linewidth=2.5,
    ))
    ax.annotate(
        "μ, σ² over D\n(per token, per sample)",
        xy=(hx + token_w / 2, hy + token_h / 2),
        xytext=(hx + 2.2, hy + 0.9),
        fontsize=10, color=C_NORM, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_NORM, lw=2),
        ha="left",
    )

    # Axis labels
    ax.annotate("", xy=(start_x + L * token_w + 0.2, start_y - N * 0.55), xytext=(start_x + L * token_w + 0.2, start_y + token_h),
                arrowprops=dict(arrowstyle="<->", color=C_BATCH, lw=2))
    ax.text(start_x + L * token_w + 0.45, start_y - N * 0.55 / 2, "N", va="center", fontsize=10, color=C_BATCH)

    ax.annotate("", xy=(start_x + L * token_w, start_y + 0.55), xytext=(start_x, start_y + 0.55),
                arrowprops=dict(arrowstyle="<->", color=C_SPATIAL, lw=2))
    ax.text(start_x + L * token_w / 2, start_y + 0.75, "L (sequence length)", ha="center", fontsize=10, color=C_SPATIAL)

    # D zoom
    zx, zy = 6.5, 1.0
    ax.text(zx + 1.0, 2.5, "Zoom: one token vector", ha="center", fontsize=11, fontweight="bold")
    vals = rng.normal(0, 1, D)
    for d in range(D):
        color = plt.cm.RdBu_r((vals[d] + 2) / 4)
        ax.add_patch(Rectangle((zx + d * 0.28, zy), 0.25, 0.8, facecolor=color, edgecolor="#555"))
        ax.text(zx + d * 0.28 + 0.125, zy - 0.2, f"d{d}", ha="center", fontsize=7)
    ax.add_patch(FancyBboxPatch(
        (zx - 0.08, zy - 0.05), D * 0.28 + 0.08, 0.9,
        boxstyle="round,pad=0.02", facecolor=C_NORM, alpha=0.15, edgecolor=C_NORM, linewidth=2,
    ))
    ax.text(zx + D * 0.14, zy + 1.05, "LayerNorm(512): stats over these D features",
            ha="center", fontsize=10, color=C_NORM, fontweight="bold")

    legend = [
        mpatches.Patch(color=C_BATCH, label="Batch (N) — independent per row"),
        mpatches.Patch(color=C_SPATIAL, label="Sequence (L) — each token normalized separately"),
        mpatches.Patch(color=C_NORM, alpha=0.4, label="Feature dim (D) — averaged for μ, σ²"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=1, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    path = OUT_DIR / "02_layernorm_concept.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_numeric_example():
    """Small numeric before/after for both norms."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Numeric toy example (simplified)", fontsize=14, fontweight="bold", y=1.02)

    # BatchNorm: 2 samples, 2 channels, 2x2 spatial
    x_bn = np.array([
        [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],   # sample 0
        [[[2., 3.], [4., 5.]], [[6., 7.], [8., 9.]]],   # sample 1
    ])
    # Channel 0 stats over N,H,W: values 1,2,3,4,2,3,4,5
    ch0 = x_bn[:, 0].ravel()
    mu0, sig0 = ch0.mean(), ch0.std()
    ch1 = x_bn[:, 1].ravel()
    mu1, sig1 = ch1.mean(), ch1.std()

    ax = axes[0]
    ax.set_title("BatchNorm — per channel c\nstats over N × H × W", fontsize=12, fontweight="bold")
    ax.axis("off")

    txt = (
        "Input shape: (N=2, C=2, H=2, W=2)\n\n"
        f"Channel 0 values: {list(ch0.round(1))}\n"
        f"  μ₀ = {mu0:.2f},  σ₀ = {sig0:.2f}\n\n"
        f"Channel 1 values: {list(ch1.round(1))}\n"
        f"  μ₁ = {mu1:.2f},  σ₁ = {sig1:.2f}\n\n"
        "Each channel gets its own\nnormalization (shared γ_c, β_c).\n"
        "Other samples affect your stats."
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10,
            family="monospace", bbox=dict(boxstyle="round", facecolor="#E8F0FE", alpha=0.9))

    # LayerNorm: 2 samples, 3 features
    x_ln = np.array([
        [1., 2., 9.],
        [4., 5., 6.],
    ])
    ax = axes[1]
    ax.set_title("LayerNorm — per sample n\nstats over D features", fontsize=12, fontweight="bold")
    ax.axis("off")

    lines = ["Input shape: (N=2, D=3)\n"]
    for n in range(2):
        row = x_ln[n]
        mu, sig = row.mean(), row.std()
        norm = (row - mu) / (sig + 1e-5)
        lines.append(f"Sample {n}: {list(row.astype(int))}")
        lines.append(f"  μ = {mu:.2f}, σ = {sig:.2f}")
        lines.append(f"  normalized ≈ {list(norm.round(2))}\n")
    lines.append("Each sample normalized alone.\nBatch mates don't matter.")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10,
            family="monospace", bbox=dict(boxstyle="round", facecolor="#FFF0E8", alpha=0.9))

    fig.tight_layout()
    path = OUT_DIR / "03_numeric_example.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_side_by_side_summary():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 5.6, "Which dimensions are averaged?", ha="center", fontsize=15, fontweight="bold")

    # BatchNorm panel
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 4.4, 4.8, boxstyle="round,pad=0.08",
                                facecolor="#E8F0FE", edgecolor=C_BATCH, linewidth=2))
    ax.text(2.5, 4.9, "BatchNorm", ha="center", fontsize=13, fontweight="bold", color=C_BATCH)
    ax.text(2.5, 4.35, "(N, C, H, W)", ha="center", fontsize=11, family="monospace")

    bn_table = (
        "For fixed channel c:\n\n"
        "  average over →  N, H, W\n"
        "  keep separate →  C\n\n"
        "Learnable: γ_c, β_c per channel\n\n"
        "Train: batch stats\n"
        "Eval:  running mean/var\n\n"
        "Best for: CNNs, large batches"
    )
    ax.text(0.6, 3.9, bn_table, va="top", fontsize=10, family="monospace")

    # LayerNorm panel
    ax.add_patch(FancyBboxPatch((5.3, 0.4), 4.4, 4.8, boxstyle="round,pad=0.08",
                                facecolor="#FFF0E8", edgecolor=C_NORM, linewidth=2))
    ax.text(7.5, 4.9, "LayerNorm", ha="center", fontsize=13, fontweight="bold", color=C_NORM)
    ax.text(7.5, 4.35, "(N, L, D)", ha="center", fontsize=11, family="monospace")

    ln_table = (
        "For fixed sample (n, l):\n\n"
        "  average over →  D\n"
        "  keep separate →  N, L\n\n"
        "Learnable: γ, β per feature\n\n"
        "Train & eval: same formula\n"
        "No running statistics\n\n"
        "Best for: Transformers, seq models"
    )
    ax.text(5.6, 3.9, ln_table, va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    path = OUT_DIR / "04_side_by_side_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    paths = [
        fig_batchnorm_concept(),
        fig_layernorm_concept(),
        fig_numeric_example(),
        fig_side_by_side_summary(),
    ]
    print("Saved figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
