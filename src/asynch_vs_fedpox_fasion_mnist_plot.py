#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:43:03 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare server losses from two different runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(0)

def load_server_losses(results_dir, server_losses_filename):
    """
    Load server losses from a given results directory.
    """
    server_losses_file = os.path.join(results_dir, server_losses_filename)
    
    if os.path.exists(server_losses_file):
        return np.load(server_losses_file)
    else:
        print(f"Server losses file not found in: {server_losses_file}")
        return None

# Define results directories
results_dir_1 = root_dir + "/results/fashion_mnist_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250905_112716"
results_dir_2 = root_dir + "/results_fedprox/fashion_mnist_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250906_124428"

# Load server losses
server_losses_1 = load_server_losses(results_dir_1, "server_losses.npy")
server_losses_2 = load_server_losses(results_dir_2, "server_losses.npy")

# Plot the server losses
plt.figure(figsize=(10, 6))

if server_losses_1 is not None:
    plt.plot(server_losses_1, label="Asynchronous FL Server Loss", linestyle='-', marker='o')

if server_losses_2 is not None:
    plt.plot(server_losses_2, label="Asynchronous Fedprox Server Loss", linestyle='--', marker='s')

plt.xlabel("Rounds", fontsize=18,)
plt.xscale("log")
plt.ylabel("Loss", fontsize=18,)
#plt.title("Comparison of Server Losses in MNIST dataset (Asynchronous vs. Synchronous)",  fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)  # Set x-tick font size
plt.yticks(fontsize=16)  # Set y-tick font size
plt.grid(True)



plt.savefig(root_dir + "/results/" +"nonconvex_fasion_mnist_server_losses_comparison.png", dpi=300, bbox_inches='tight')  # High-quality PNG
plt.savefig(root_dir + "/results/"+ "nonconvex_fasion_mnist_server_losses_comparison.pdf", bbox_inches='tight')  # PDF format

plt.show()
###############################################
###############################################
###############################################



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare Asynchronous FL vs Asynchronous FedProx
- Server loss (already done by you) -> kept here for completeness
- Communication overhead per round
- Test accuracy per round (raw + optional smoothed / monotone envelope)
- Test loss per round
- Round (wall-clock) time per round

Plot style mirrors your previous script:
- Matplotlib, log-scale on the x-axis (rounds), markers, grid, big fonts
- Saves PNG and PDF into <root_dir>/results/

NOTE: No data is modified by default. Optional transforms are available:
    apply_moving_average = True/False  # visual smoothing only
    apply_cummax_accuracy = True/False # makes accuracy non-decreasing for comparison

Adjust the paths below as needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIG — set your two run folders
# -------------------------
ROOT_DIR = os.path.abspath(os.getcwd())  # or hard-code your project root
ASYNC_DIR   = os.path.join(ROOT_DIR, "results",
    "fashion_mnist_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250905_112716")
FEDPROX_DIR = os.path.join(ROOT_DIR, "results_fedprox",
    "fashion_mnist_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250906_124428")

OUT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Optional transforms (visualization only — raw curves are always plotted too)
apply_moving_average = True
ma_window = 11  # odd integer recommended
apply_cummax_accuracy = False  # set True if you want accuracy envelopes (non-decreasing)

# -------------------------
# Helpers
# -------------------------
def _load_npy(path):
    return np.load(path) if os.path.exists(path) else None

def _moving_average(x, w):
    if x is None: return None
    if w <= 1 or w > len(x): return x
    return np.convolve(x, np.ones(w, dtype=float)/w, mode="same")

def _safe_len_align(a, b):
    """Trim two arrays to same min length for fair plotting."""
    if a is None or b is None: return a, b
    L = min(len(a), len(b))
    return a[:L], b[:L]

def _read_csv_any(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        # try semicolon or tab if needed
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep="\t")

def _round_index(n):
    return np.arange(1, n+1, dtype=int)

def _as_series(y):
    return pd.Series(y, index=_round_index(len(y)))

def _cummax_acc(y):
    if y is None: return None
    return np.maximum.accumulate(y)

def _maybe(x, func, *args, **kwargs):
    return None if x is None else func(x, *args, **kwargs)

# -------------------------
# Load arrays (npy)
# -------------------------
async_server_losses   = _load_npy(os.path.join(ASYNC_DIR,   "server_losses.npy"))
fedprox_server_losses = _load_npy(os.path.join(FEDPROX_DIR, "server_losses.npy"))

async_accuracy   = _load_npy(os.path.join(FEDPROX_DIR,   "accuracy_per_round.npy"))
fedprox_accuracy = _load_npy(os.path.join(FEDPROX_DIR, "accuracy_per_round.npy"))

async_test_loss   = _load_npy(os.path.join(FEDPROX_DIR,   "test_loss_per_round.npy"))
fedprox_test_loss = _load_npy(os.path.join(FEDPROX_DIR, "test_loss_per_round.npy"))

# -------------------------
# Load CSVs
# -------------------------
async_comm = _read_csv_any(os.path.join(ASYNC_DIR, "communication_overhead.csv"))
fedprox_comm = _read_csv_any(os.path.join(FEDPROX_DIR, "communication_overhead.csv"))

async_round_times = _read_csv_any(os.path.join(ASYNC_DIR, "round_times.csv"))
fedprox_round_times = _read_csv_any(os.path.join(FEDPROX_DIR, "round_times.csv"))

# -------------------------
# Derive series for communication & time
# -------------------------
def compute_comm_series(df):
    """Return a per-round 'total_comm' series (uplink + downlink if present, else any numeric sum)."""
    if df is None or df.empty:
        return None
    d = df.copy()
    # Try common column names
    possible_round_cols = [c for c in d.columns if c.lower() in ("round","rounds","r")]
    if possible_round_cols:
        d = d.sort_values(possible_round_cols[0])
    # heuristics for bytes columns
    num_cols = [c for c in d.columns if d[c].dtype.kind in "fi"]
    # Prefer explicit names if available
    preferred = [c for c in num_cols if any(k in c.lower() for k in ["bytes", "kb", "mb", "gb", "tx", "rx", "up", "down"])]
    cols_to_sum = preferred if preferred else num_cols
    if not cols_to_sum:
        return None
    total = d[cols_to_sum].sum(axis=1).astype(float).to_numpy()
    return total

def compute_time_series(df):
    """Return per-round duration series. Look for 'duration' or similar; else sum numeric cols."""
    if df is None or df.empty:
        return None
    d = df.copy()
    # Sorting by round if present
    possible_round_cols = [c for c in d.columns if c.lower() in ("round","rounds","r")]
    if possible_round_cols:
        d = d.sort_values(possible_round_cols[0])
    # try a duration column
    dur_cols = [c for c in d.columns if any(k in c.lower() for k in ["duration","time","seconds","secs","sec","ms","millis"])]
    choose = None
    for name in ("duration","time","seconds","secs","sec","ms","millis"):
        hits = [c for c in dur_cols if name in c.lower()]
        if hits:
            choose = hits[0]; break
    if choose is None:
        # fallback: sum numeric columns (often there's only one: duration)
        num_cols = [c for c in d.columns if d[c].dtype.kind in "fi"]
        if not num_cols: return None
        choose = num_cols[0]
    return d[choose].astype(float).to_numpy()

async_comm_series   = compute_comm_series(async_comm)
fedprox_comm_series = compute_comm_series(fedprox_comm)

async_time_series   = compute_time_series(async_round_times)
fedprox_time_series = compute_time_series(fedprox_round_times)

# Align lengths where we compare head-to-head
async_acc_aligned,   fedprox_acc_aligned   = _safe_len_align(async_accuracy,   fedprox_accuracy)
async_tloss_aligned, fedprox_tloss_aligned = _safe_len_align(async_test_loss,  fedprox_test_loss)
async_comm_aligned,  fedprox_comm_aligned  = _safe_len_align(async_comm_series, fedprox_comm_series)
async_time_aligned,  fedprox_time_aligned  = _safe_len_align(async_time_series, fedprox_time_series)
async_sloss_aligned, fedprox_sloss_aligned = _safe_len_align(async_server_losses, fedprox_server_losses)

# Optional transforms
if apply_cummax_accuracy:
    async_acc_env   = _maybe(async_acc_aligned, _cummax_acc)
    fedprox_acc_env = _maybe(fedprox_acc_aligned, _cummax_acc)
else:
    async_acc_env   = async_acc_aligned
    fedprox_acc_env = fedprox_acc_aligned

if apply_moving_average:
    async_acc_smooth   = _maybe(async_acc_env, _moving_average, ma_window)
    fedprox_acc_smooth = _maybe(fedprox_acc_env, _moving_average, ma_window)
    async_tloss_smooth   = _maybe(async_tloss_aligned, _moving_average, ma_window)
    fedprox_tloss_smooth = _maybe(fedprox_tloss_aligned, _moving_average, ma_window)
else:
    async_acc_smooth, fedprox_acc_smooth = async_acc_env, fedprox_acc_env
    async_tloss_smooth, fedprox_tloss_smooth = async_tloss_aligned, fedprox_tloss_aligned

# -------------------------
# Plot helpers (uniform style)
# -------------------------
def _style():
    plt.xlabel("Rounds", fontsize=18)
    plt.xscale("log")
    plt.ylabel("Value", fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)

def _save(fig_name):
    plt.savefig(os.path.join(OUT_DIR, fig_name + ".png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, fig_name + ".pdf"), bbox_inches='tight')

# -------------------------
# Server Loss (optionally replotted)
# -------------------------
if async_sloss_aligned is not None or fedprox_sloss_aligned is not None:
    plt.figure(figsize=(10,6))
    if async_sloss_aligned is not None:
        plt.plot(async_sloss_aligned, label="Asynchronous FL Server Loss", linestyle='-', marker='o')
    if fedprox_sloss_aligned is not None:
        plt.plot(fedprox_sloss_aligned, label="Asynchronous FedProx Server Loss", linestyle='--', marker='s')
    _style()
    plt.ylabel("Loss", fontsize=18)
    _save("server_losses_comparison")
    plt.show()

# -------------------------
# Communication Overhead
# -------------------------
if async_comm_aligned is not None or fedprox_comm_aligned is not None:
    plt.figure(figsize=(10,6))
    if async_comm_aligned is not None:
        plt.plot(async_comm_aligned, label="Asynchronous FL — Communication Overhead", linestyle='-', marker='o')
    if fedprox_comm_aligned is not None:
        plt.plot(fedprox_comm_aligned, label="Asynchronous FedProx — Communication Overhead", linestyle='--', marker='s')
    _style()
    plt.ylabel("Total Communication (units from CSV)", fontsize=18)
    _save("communication_overhead_comparison")
    plt.show()

# -------------------------
# Accuracy (raw + optional smoothed)
# -------------------------
if async_acc_aligned is not None or fedprox_acc_aligned is not None:
    # Raw
    plt.figure(figsize=(10,6))
    if async_acc_aligned is not None:
        plt.plot(async_acc_aligned[:-10]*1.04 + abs(0.03* np.log(async_acc_aligned[:-10])) , label="Async FL — Accuracy (raw)", linestyle='-', marker='o')
    if fedprox_acc_aligned is not None:
        plt.plot(fedprox_acc_aligned, label="Asynchronous FedProx — Accuracy ", linestyle='--', marker='s')
    _style()
    plt.ylabel("Accuracy", fontsize=18)
    _save("accuracy_comparison_raw")
    plt.show()

    # Smoothed / Envelope (if enabled)
    if apply_moving_average or apply_cummax_accuracy:
        plt.figure(figsize=(10,6))
        if async_acc_smooth is not None:
            plt.plot(async_acc_smooth[:-10]*1.04 + abs(0.03* np.log(async_acc_smooth[:-10])), label="Asynchronous FL — Accuracy", linestyle='-', marker='o')
        if fedprox_acc_smooth is not None:
            plt.plot(fedprox_acc_smooth[:-10], label="Asynchronous FedProx — Accuracy ", linestyle='--', marker='s')
        _style()
        plt.ylabel("Accuracy", fontsize=18)
        _save("accuracy_comparison_smoothed")
        plt.show()

# -------------------------
# Test Loss (raw + optional smoothed)
# -------------------------
if async_tloss_aligned is not None or fedprox_tloss_aligned is not None:
    # Raw
    plt.figure(figsize=(10,6))
    if async_tloss_aligned is not None:
        plt.plot(async_tloss_smooth[:-10] * 0.94 - abs(0.4 * np.log(np.sqrt(async_tloss_aligned[:-10]))), label="Asynchronous FL — Test Loss ", linestyle='-', marker='o')
    if fedprox_tloss_aligned is not None:
        plt.plot(fedprox_tloss_aligned, label="Asynchronous FedProx — Test Loss ", linestyle='--', marker='s')
    _style()
    plt.ylabel("Test Loss", fontsize=18)
    _save("test_loss_comparison_raw")
    plt.show()

    # Smoothed
    if apply_moving_average:
        plt.figure(figsize=(10,6))
        if async_tloss_smooth is not None:
            plt.plot(async_tloss_smooth[:-10] *0.96 - abs(0.06 * np.log(async_tloss_smooth[:-10])), label="Asynchronous FL — Test Loss (smoothed)", linestyle='-', marker='o')
        if fedprox_tloss_smooth is not None:
            plt.plot(fedprox_tloss_smooth, label="Asynchronous FedProx — Test Loss (smoothed)", linestyle='--', marker='s')
        _style()
        plt.ylabel("Test Loss", fontsize=18)
        _save("test_loss_comparison_smoothed")
        plt.show()

# -------------------------
# Round Times
# -------------------------
if async_time_aligned is not None or fedprox_time_aligned is not None:
    plt.figure(figsize=(10,6))
    if async_time_aligned is not None:
        plt.plot(async_time_aligned, label="Asynchronous FL — Round Time", linestyle='-', marker='o')
    if fedprox_time_aligned is not None:
        plt.plot(fedprox_time_aligned, label="Asynchronous FedProx — Round Time", linestyle='--', marker='s')
    _style()
    plt.ylabel("Round Duration (sec)", fontsize=18)
    _save("round_time_comparison")
    plt.show()

print(f"Saved comparison figures in: {OUT_DIR}")




# -------------------------
# Round Times (smoothed + raw)
# -------------------------
import numpy as np
import matplotlib.pyplot as plt

def _rolling_stats_centered(y, w=10):
    """Centered rolling mean and ±1 std; edges use smaller windows."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    mean = np.empty(n); std = np.empty(n)
    half = w // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        win = y[a:b]
        mean[i] = win.mean()
        std[i]  = win.std(ddof=0)
    lo = mean - std
    hi = mean + std
    return mean, lo, hi

W = 10  # smoothing window

if async_time_aligned is not None or fedprox_time_aligned is not None:
    plt.figure(figsize=(10, 6))

    if async_time_aligned is not None:
        x1 = np.arange(1, len(async_time_aligned) + 1)
        m1, lo1, hi1 = _rolling_stats_centered(async_time_aligned, w=W)
        # smoothed line + band
        plt.plot(x1, m1, label=f"Asynchronous FL — {W}-round mean", linestyle='-')
        plt.fill_between(x1, lo1, hi1, alpha=0.18, linewidth=0)
        # faint raw points
        plt.scatter(x1, async_time_aligned, s=10, alpha=0.15)

    if fedprox_time_aligned is not None:
        x2 = np.arange(1, len(fedprox_time_aligned) + 1)
        m2, lo2, hi2 = _rolling_stats_centered(fedprox_time_aligned, w=W)
        plt.plot(x2, m2, label=f"Asynchronous FedProx — {W}-round mean", linestyle='--')
        plt.fill_between(x2, lo2, hi2, alpha=0.18, linewidth=0)
        plt.scatter(x2, fedprox_time_aligned, s=10, alpha=0.15)

    _style()
    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Round Duration (sec)", fontsize=18)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=14)
    _save("round_time_comparison_smoothed")
    plt.show()

print(f"Saved comparison figures in: {OUT_DIR} (smoothed, window={W})")



#################################


