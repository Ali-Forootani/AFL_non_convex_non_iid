#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 12:34:16 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate communication overhead for two async methods (no re-training required).

Async FL:
  - Broadcasts quantized server weights each round (downlink compression).
  - Uplink uses top-k sparsification; sparsity increases over rounds (payload shrinks).
  - Occasional client unavailability reduces selected client count.

Async FedProx:
  - Same downlink as FL.
  - Uplink is less sparse (prox term -> denser deltas) and may upload once per local epoch
    with small probability (extra traffic).
  - Slightly higher retry probability.

Outputs:
  - Per-round communication (MB) for both methods
  - Cumulative communication (MB)
  - Summary stats
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config (tweak as desired)
# -------------------------
R = 1000                   # rounds
avg_clients = 5            # target selected clients per round
max_clients = 10
rng = np.random.default_rng(42)

# "Model size" if sent in full precision without compression (MB)
model_mb = 5.0             # ~ a small CNN; change as you like

# Downlink compression (e.g., 8-bit quantization ~ 0.25 of fp32)
downlink_ratio = 0.35      # fraction of full model size broadcast per client

# Availability: number of clients selected each round varies
def draw_selected_clients():
    # clip a Poisson around avg_clients between 1 and max_clients
    c = np.clip(rng.poisson(lam=avg_clients), 1, max_clients)
    return int(c)

# Retry probabilities (resends add extra traffic)
retry_p_fl = 0.02
retry_p_prox = 0.04

# Uplink sparsity schedules (fraction of elements zeroed)
# Start less sparse and get sparser as training progresses
def sparsity_schedule_fl(t, T=R):
    # goes from ~0.30 to ~0.85 across training
    return 0.30 + 0.55 * (t / (T - 1))

def sparsity_schedule_prox(t, T=R):
    # FedProx typically a bit denser updates (less sparse): ~0.20 -> ~0.75
    return 0.20 + 0.55 * (t / (T - 1))

# Index overhead for top-k (very rough; indices + values)
index_overhead_per_element = 8.0 / (1024.0 * 1024.0)  # 8 bytes per element, expressed in MB

# Occasional per-epoch uploads in FedProx (extra uplink bursts)
local_epochs = 10
per_epoch_upload_prob = 0.05   # with this prob, a client also uploads once per epoch

# -------------------------
# Simulation
# -------------------------
def simulate_round(method, t):
    """
    Return (per_round_MB, per_round_breakdown_dict)
    """
    C = draw_selected_clients()

    # Downlink: broadcast to C clients with compression
    down_mb = C * (model_mb * downlink_ratio)

    # Uplink: depends on method
    if method == "fl":
        s = sparsity_schedule_fl(t)
        # fraction kept is (1 - s); we also add a small index overhead for kept elements
        uplink_per_client = model_mb * (1.0 - s) + model_mb * (1.0 - s) * 0.02  # +2% meta
        uplink_mb = C * uplink_per_client

        # occasional retries
        if rng.random() < retry_p_fl:
            uplink_mb *= 1.2  # +20% retransmit
            down_mb   *= 1.05 # small extra control msgs etc.

    elif method == "fedprox":
        s = sparsity_schedule_prox(t)
        uplink_per_client = model_mb * (1.0 - s) + model_mb * (1.0 - s) * 0.03  # +3% meta
        uplink_mb = C * uplink_per_client

        # occasional per-epoch uploads (rare but heavy)
        if rng.random() < per_epoch_upload_prob:
            # upload lightweight summaries each epoch (e.g., compressed grads)
            uplink_mb += C * local_epochs * (model_mb * 0.02)  # 2% of model per epoch

        # retries more likely
        if rng.random() < retry_p_prox:
            uplink_mb *= 1.25
            down_mb   *= 1.05
    else:
        raise ValueError("Unknown method")

    total_mb = down_mb + uplink_mb
    return total_mb, {"C": C, "down_mb": down_mb, "uplink_mb": uplink_mb}

# Run simulation
fl_per_round = np.zeros(R)
prox_per_round = np.zeros(R)
for t in range(R):
    fl_per_round[t], _ = simulate_round("fl", t)
    prox_per_round[t], _ = simulate_round("fedprox", t)

# Cumulative
fl_cum = np.cumsum(fl_per_round)
prox_cum = np.cumsum(prox_per_round)

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,6))
plt.plot(fl_per_round, label="Async FL — per-round MB", linewidth=1)
plt.plot(prox_per_round, label="Async FedProx — per-round MB", linewidth=1, linestyle="--")
plt.xscale("log")
plt.xlabel("Rounds", fontsize=16)
plt.ylabel("Per-round Communication (MB)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(fl_cum, label="Async FL — cumulative MB", linewidth=2)
plt.plot(prox_cum, label="Async FedProx — cumulative MB", linewidth=2, linestyle="--")
plt.xscale("log")
plt.xlabel("Rounds", fontsize=16)
plt.ylabel("Cumulative Communication (MB)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# -------------------------
# Summary
# -------------------------
def summarize(name, x):
    print(f"{name}: median={np.median(x):.2f} MB/round | p90={np.percentile(x,90):.2f} MB/round | total={np.sum(x):.1f} MB")

summarize("Async FL     ", fl_per_round)
summarize("Async FedProx", prox_per_round)





#####################################################


import os

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


def _save(fig_name):
    plt.savefig(os.path.join(OUT_DIR, fig_name + ".png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, fig_name + ".pdf"), bbox_inches='tight')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate communication overhead for two async methods and plot
10-round centered rolling mean with ±1 std bands.

Outputs:
  - Per-round MB (raw + smoothed + band)
  - Cumulative MB
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config (tweak as desired)
# -------------------------
R = 1000                   # rounds
avg_clients = 5            # target selected clients per round
max_clients = 10
rng = np.random.default_rng(42)

# "Model size" if sent in full precision without compression (MB)
model_mb = 5.0             # ~ a small CNN

# Downlink compression (e.g., quantization)
downlink_ratio = 0.35      # fraction of full model size broadcast per client

# Availability: number of clients selected each round varies
def draw_selected_clients():
    return int(np.clip(rng.poisson(lam=avg_clients), 1, max_clients))

# Retry probabilities (resends add extra traffic)
retry_p_fl = 0.02
retry_p_prox = 0.04

# Uplink sparsity schedules (fraction zeroed)
def sparsity_schedule_fl(t, T=R):     # ~0.30 -> ~0.85
    return 0.30 + 0.55 * (t / (T - 1))

def sparsity_schedule_prox(t, T=R):   # ~0.20 -> ~0.75
    return 0.20 + 0.55 * (t / (T - 1))

# Occasional per-epoch uploads in FedProx (extra uplink bursts)
local_epochs = 10
per_epoch_upload_prob = 0.05   # with this prob, upload once per epoch (compressed)

# -------------------------
# Simulation
# -------------------------
def simulate_round(method, t):
    C = draw_selected_clients()
    down_mb = C * (model_mb * downlink_ratio)

    if method == "fl":
        s = sparsity_schedule_fl(t)
        uplink_per_client = model_mb * (1.0 - s) * 1.02  # +2% meta
        uplink_mb = C * uplink_per_client
        if rng.random() < retry_p_fl:
            uplink_mb *= 1.2
            down_mb   *= 1.05

    elif method == "fedprox":
        s = sparsity_schedule_prox(t)
        uplink_per_client = model_mb * (1.0 - s) * 1.03  # +3% meta
        uplink_mb = C * uplink_per_client
        if rng.random() < per_epoch_upload_prob:
            uplink_mb += C * local_epochs * (model_mb * 0.02)
        if rng.random() < retry_p_prox:
            uplink_mb *= 1.25
            down_mb   *= 1.05
    else:
        raise ValueError("Unknown method")

    total_mb = down_mb + uplink_mb
    return total_mb

fl_per_round   = np.array([simulate_round("fl", t)      for t in range(R)])
prox_per_round = np.array([simulate_round("fedprox", t) for t in range(R)])

fl_cum   = np.cumsum(fl_per_round)
prox_cum = np.cumsum(prox_per_round)

# -------------------------
# Centered rolling stats (10-round)
# -------------------------
def rolling_mean_std_centered(y, w=10):
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
    return mean, std

W = 10
fl_mean,   fl_std   = rolling_mean_std_centered(fl_per_round,   w=W)
prox_mean, prox_std = rolling_mean_std_centered(prox_per_round, w=W)

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,6))
x = np.arange(1, R+1)

# Async FL
plt.plot(x, fl_mean, label=f"Asynchronous  FL — {W}-round mean", linewidth=2)
plt.fill_between(x, fl_mean - fl_std, fl_mean + fl_std, alpha=0.18, linewidth=0)
plt.scatter(x, fl_per_round, s=8, alpha=0.12)

# Async FedProx
plt.plot(x, prox_mean, label=f"Asynchronous  FedProx — {W}-round mean", linestyle="--", linewidth=2)
plt.fill_between(x, prox_mean - prox_std, prox_mean + prox_std, alpha=0.18, linewidth=0)
plt.scatter(x, prox_per_round, s=8, alpha=0.12)

plt.xscale("log")
plt.xlabel("Rounds", fontsize=16)
plt.ylabel("Per-round Communication (MB)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
_save("communication_overhead")
plt.show()

# Cumulative MB (for completeness)
plt.figure(figsize=(10,6))
plt.plot(x, fl_cum,   label="Asynchronous  FL — cumulative MB", linewidth=2)
plt.plot(x, prox_cum, label="Asynchronous  FedProx — cumulative MB", linestyle="--", linewidth=2)
plt.xscale("log")
plt.xlabel("Rounds", fontsize=16)
plt.ylabel("Cumulative Communication (MB)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()


















