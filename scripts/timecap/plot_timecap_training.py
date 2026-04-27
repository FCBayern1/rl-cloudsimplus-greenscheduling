#!/usr/bin/env python3
"""
Parse a TimeCAP training log file and plot all training-curve metrics.

Metrics extracted (whatever the log contains):
  - Train Loss / Vali Loss / Test Loss per epoch
  - Time spent per epoch (seconds)
  - Learning rate per epoch
  - Autoregressive loss / One-shot loss (intermediate validation prints)
  - Final test MSE / MAE
  - Best-saved epoch (where validation loss decreased)

Usage:
    python plot_timecap_training.py LOG_FILE [-o OUTPUT_DIR]

Example:
    python plot_timecap_training.py logs/timecap_train_4358062.out
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(
    r"Epoch:\s*(\d+)\s+Spend:\s*(\d+)\s*s\s*\|\s*"
    r"Train Loss:\s*([\d.eE+-]+)\s+"
    r"Vali Loss:\s*([\d.eE+-]+)\s+"
    r"Test Loss:\s*([\d.eE+-]+)"
)
LR_RE = re.compile(r"Updating learning rate to\s*([\d.eE+-]+)")
SAVE_RE = re.compile(r"Saving model of epoch\s*(\d+)")
AR_RE = re.compile(r"Autoregressive loss:\s*([\d.eE+-]+)")
OS_RE = re.compile(r"One-shot loss:\s*([\d.eE+-]+)")
FINAL_RE = re.compile(r"MSE:\s*([\d.eE+-]+)[,\s]+MAE:\s*([\d.eE+-]+)")


def parse_log(path: Path):
    epochs, spend, train_l, vali_l, test_l = [], [], [], [], []
    saved_epochs = []
    ar_losses, os_losses = [], []
    lrs_after_epoch = []  # only the lr printed once per epoch (after Saving)
    final_mse, final_mae = None, None

    # First pass: per-line scan
    with path.open("r", errors="replace") as f:
        for raw in f:
            # tqdm uses \r — split on it so we can scan inner lines too
            for line in raw.split("\r"):
                line = line.strip()
                if not line:
                    continue

                m = EPOCH_RE.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    spend.append(int(m.group(2)))
                    train_l.append(float(m.group(3)))
                    vali_l.append(float(m.group(4)))
                    test_l.append(float(m.group(5)))
                    continue

                m = SAVE_RE.search(line)
                if m:
                    saved_epochs.append(int(m.group(1)))
                    continue

                m = AR_RE.search(line)
                if m:
                    ar_losses.append(float(m.group(1)))
                    # don't continue — same line could in theory have one-shot
                m = OS_RE.search(line)
                if m:
                    os_losses.append(float(m.group(1)))

                m = LR_RE.search(line)
                if m:
                    lrs_after_epoch.append(float(m.group(1)))

                m = FINAL_RE.search(line)
                if m:
                    final_mse = float(m.group(1))
                    final_mae = float(m.group(2))

    # The log prints "Updating learning rate" many times (once per rank /
    # repeatedly). De-duplicate by collapsing consecutive equal values into one.
    lr_per_epoch = []
    for v in lrs_after_epoch:
        if not lr_per_epoch or lr_per_epoch[-1] != v:
            lr_per_epoch.append(v)

    return {
        "epochs": np.array(epochs),
        "spend": np.array(spend),
        "train_loss": np.array(train_l),
        "vali_loss": np.array(vali_l),
        "test_loss": np.array(test_l),
        "saved_epochs": sorted(set(saved_epochs)),
        "ar_loss": np.array(ar_losses),
        "os_loss": np.array(os_losses),
        "lr": np.array(lr_per_epoch),
        "final_mse": final_mse,
        "final_mae": final_mae,
    }


def best_epoch(d):
    if len(d["vali_loss"]) == 0:
        return None
    idx = int(np.argmin(d["vali_loss"]))
    return int(d["epochs"][idx]), float(d["vali_loss"][idx])


def plot_all(d: dict, log_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem

    # Decide subplot layout based on what we actually have
    panels = []
    if len(d["epochs"]):
        panels.append("losses")
        panels.append("loss_log")
        panels.append("spend")
    if len(d["lr"]):
        panels.append("lr")
    if len(d["ar_loss"]) or len(d["os_loss"]):
        panels.append("ar_os")

    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    be = best_epoch(d)

    for ax, panel in zip(axes, panels):
        if panel == "losses":
            ax.plot(d["epochs"], d["train_loss"], "-o", label="Train", ms=4)
            ax.plot(d["epochs"], d["vali_loss"], "-s", label="Vali", ms=4)
            ax.plot(d["epochs"], d["test_loss"], "-^", label="Test", ms=4)
            if be:
                ax.axvline(be[0], color="red", ls="--", lw=1,
                           label=f"best epoch={be[0]}")
            ax.set_title("Loss per Epoch (linear)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            ax.legend()

        elif panel == "loss_log":
            ax.plot(d["epochs"], d["train_loss"], "-o", label="Train", ms=4)
            ax.plot(d["epochs"], d["vali_loss"], "-s", label="Vali", ms=4)
            ax.plot(d["epochs"], d["test_loss"], "-^", label="Test", ms=4)
            ax.set_yscale("log")
            ax.set_title("Loss per Epoch (log scale)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.grid(alpha=0.3, which="both")
            ax.legend()

        elif panel == "spend":
            ax.bar(d["epochs"], d["spend"] / 60.0, color="steelblue")
            ax.set_title("Time per Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Minutes")
            mean_min = float(np.mean(d["spend"]) / 60.0)
            ax.axhline(mean_min, color="red", ls="--", lw=1,
                       label=f"mean ≈ {mean_min:.1f} min")
            ax.grid(alpha=0.3, axis="y")
            ax.legend()

        elif panel == "lr":
            xs = np.arange(1, len(d["lr"]) + 1)
            ax.plot(xs, d["lr"], "-o", color="darkorange", ms=4)
            ax.set_yscale("log")
            ax.set_title("Learning Rate Schedule")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR (log)")
            ax.grid(alpha=0.3, which="both")

        elif panel == "ar_os":
            if len(d["ar_loss"]):
                ax.plot(d["ar_loss"], label="Autoregressive", lw=1)
            if len(d["os_loss"]):
                ax.plot(d["os_loss"], label="One-shot", lw=1)
            ax.set_title("Validation-step Losses (raw prints)")
            ax.set_xlabel("Print index (across training)")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            ax.legend()

    # Hide any leftover axes
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    title = f"TimeCAP training curves — {log_path.name}"
    if d["final_mse"] is not None and d["final_mae"] is not None:
        title += f"   |   final MSE={d['final_mse']:.4f}  MAE={d['final_mae']:.4f}"
    if be:
        title += f"   |   best vali={be[1]:.4f} @ epoch {be[0]}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"{stem}_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def print_summary(d):
    print("=" * 60)
    print(f"Epochs parsed     : {len(d['epochs'])}")
    if len(d["epochs"]):
        print(f"Train loss range  : {d['train_loss'].min():.4f} → "
              f"{d['train_loss'].max():.4f}")
        print(f"Vali  loss range  : {d['vali_loss'].min():.4f} → "
              f"{d['vali_loss'].max():.4f}")
        print(f"Test  loss range  : {d['test_loss'].min():.4f} → "
              f"{d['test_loss'].max():.4f}")
        print(f"Avg time/epoch    : {np.mean(d['spend'])/60:.2f} min")
        be = best_epoch(d)
        if be:
            print(f"Best epoch (vali) : {be[0]} (vali_loss={be[1]:.6f})")
        print(f"Saved checkpoints : {d['saved_epochs']}")
    if d["final_mse"] is not None:
        print(f"Final MSE / MAE   : {d['final_mse']:.4f} / {d['final_mae']:.4f}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help="Path to timecap_train_*.out log")
    p.add_argument("-o", "--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "plots",
                   help="Where to save the figure (default: <script_dir>/plots)")
    args = p.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"Log file not found: {args.log}")

    data = parse_log(args.log)
    print_summary(data)
    plot_all(data, args.log, args.out_dir)


if __name__ == "__main__":
    main()
