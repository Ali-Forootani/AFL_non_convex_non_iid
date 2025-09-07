#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 12:34:11 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Federated Learning — FedProx (non-synchronous)

- Same spirit as your async FedAvg script (per-round random client subset; no global barrier).
- Only change is on the CLIENT objective:
    L_prox(w) = task_loss(w; x, y) + (mu/2) * || w - w_t ||^2
  where w_t are server weights snapshotted at the start of the round.

Artifacts saved (per round “r”):
- Test accuracy/loss (accuracy_per_round.npy, test_loss_per_round.npy + plots)
- Server proxy loss from aggregated client updates (server_losses.npy + plot)
- Selected clients per round (selected_clients.csv + scatter)
- Per-aggregation exec times (execution_times.csv), round durations (round_times.csv)
- Communication overhead per round (communication_overhead.csv)
- Summary (summary.json)

Default dataset: Fashion-MNIST
"""

import os
import time
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import asyncio
import nest_asyncio

# -------------------------
# Config
# -------------------------
DATASET = "fashion_mnist"      # options: "fashion_mnist", "svhn", "emnist_byclass"

num_clients = 10
alpha_dirichlet = 0.5          # non-IID control (lower = more skewed)
batch_size = 64
num_clients_per_round = 5
num_rounds = 1000
local_epochs = 10

gamma_0 = 1e-3                 # base LR in client
alpha_staleness = 0.01         # LR dampening factor with delay
delay_t = 2                    # max simulated network delay in seconds (uniform[0, delay_t])
accumulation_steps = 1
early_stopping_patience = 10

mu_prox = 0.01                 # FedProx proximal strength (try {0.001, 0.01, 0.05, 0.1})

base_results_dir = "results_fedprox"

# Allow nested event loops for asyncio
nest_asyncio.apply()

# -------------------------
# Utilities (IO/plots)
# -------------------------
def create_directory(num_clients, num_rounds, local_epochs, max_clients_per_round, dataset_name, base_dir="results_fedprox"):
    dir_name = (f"{base_dir}/{dataset_name}_clients_{num_clients}_rounds_{num_rounds}"
                f"_epochs_{local_epochs}_clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_losses_as_numpy(losses, filename):
    np.save(filename, np.array(losses))

def save_selected_clients(selected_clients, filename):
    with open(filename, "w") as f:
        f.write("Round,SelectedClients\n")
        for round_num, clients in enumerate(selected_clients):
            f.write(f"{round_num},{' '.join(map(str, clients))}\n")

def save_execution_times(execution_times_by_round, filename):
    with open(filename, "w") as f:
        f.write("Round,ClientIndex,ExecutionTimeSeconds\n")
        for round_num, times in enumerate(execution_times_by_round):
            for client_idx, exec_time in enumerate(times):
                f.write(f"{round_num},{client_idx},{exec_time:.6f}\n")

def save_round_times(round_times, filename):
    with open(filename, "w") as f:
        f.write("Round,DurationSeconds\n")
        for r, dur in enumerate(round_times):
            f.write(f"{r},{dur:.6f}\n")

def save_comm_overhead(comm_rows, filename):
    with open(filename, "w") as f:
        f.write("Round,NumSelected,BytesDown,BytesUp,BytesTotal,MBTotal\n")
        for row in comm_rows:
            f.write(f"{row['round']},{row['num_clients']},{row['bytes_down']},{row['bytes_up']},"
                    f"{row['bytes_total']},{row['mb_total']:.6f}\n")

def plot_series(values, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(values, label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def state_dict_nbytes(state_dict):
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += v.nelement() * v.element_size()
    return total

# -------------------------
# Model
# -------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.zero_()
    def forward(self, x):
        return self.relu(self.conv(x))

class SimpleCNN(nn.Module):
    """
    Lightweight CNN that works for 1x28x28 (MNIST/Fashion-MNIST/EMNIST) and 3x32x32 (SVHN).
    """
    def __init__(self, input_channels=1, num_classes=10, hidden_channels=32, num_layers=3):
        super().__init__()
        self.conv_layers = nn.ModuleList([ConvLayer(input_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.conv_layers.append(ConvLayer(hidden_channels, hidden_channels))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# -------------------------
# Dataset factory
# -------------------------
def get_dataset_and_meta(name: str):
    """
    Returns: train_dataset, test_dataset, input_channels, num_classes
    """
    if name.lower() == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        return train_ds, test_ds, 1, 10

    if name.lower() == "svhn":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_ds = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        test_ds  = datasets.SVHN(root="./data", split="test",  download=True, transform=transform)
        return train_ds, test_ds, 3, 10

    if name.lower() == "emnist_byclass":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
        test_ds  = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        return train_ds, test_ds, 1, 62

    raise ValueError(f"Unsupported dataset: {name}")

# -------------------------
# Non-IID partition (Dirichlet)
# -------------------------
def partition_non_iid(dataset, num_clients, num_classes, alpha=0.5):
    """
    Partition indices by sampling class-specific proportions from a Dirichlet distribution.
    """
    data_by_class = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        data_by_class[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        cls_indices = data_by_class.get(c, [])
        if not cls_indices:
            continue
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(cls_indices)).astype(int)

        start = 0
        for i, ct in enumerate(counts):
            end = start + ct
            client_indices[i].extend(cls_indices[start:end])
            start = end
        # Any leftover due to rounding goes to random clients
        for idx_left in cls_indices[start:]:
            client_indices[np.random.randint(num_clients)].append(idx_left)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices

# -------------------------
# Client task (async) — FedProx
# -------------------------
async def train_client_fedprox(
    client_model,
    train_loader,
    device,
    local_epochs,
    loss_fn,
    gamma_0,
    alpha,
    delay_t=0,
    accumulation_steps=1,
    early_stopping_patience=10,
    server_state_at_round=None,
    mu=0.01
):
    """
    Minimize: loss(x; w) + (mu/2) * || w - w_t ||^2
    """
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []

    # Simulate network delay (asynchrony)
    await asyncio.sleep(random.uniform(0, delay_t))

    # snapshot server params for proximal term
    assert server_state_at_round is not None, "server_state_at_round cannot be None for FedProx"
    w_t = {k: v.detach().clone().to(device) for k, v in server_state_at_round.items()}

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0.0
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        optimizer = torch.optim.Adam(client_model.parameters(), lr=float(gamma_t.item()))

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.long)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = client_model(inputs)
            loss_main = loss_fn(outputs, targets)

            # proximal penalty across named parameters (match names with state_dict keys)
            prox = 0.0
            for name, p in client_model.named_parameters():
                if name in w_t:
                    prox = prox + torch.sum((p - w_t[name]) ** 2)

            loss = loss_main + (mu / 2.0) * prox
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()

            # track only task loss as proxy (for comparability)
            epoch_loss += float(loss_main.item())

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        client_losses.append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return client_model.state_dict(), client_losses

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn):
    model.eval().to(device)
    total, correct, loss_sum = 0, 0, 0.0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_sum += loss.item() * inputs.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return (loss_sum / max(1, total)), (correct / max(1, total))

# -------------------------
# Federated loop (async FedProx)
# -------------------------
async def federated_learning_fedprox(
    clients_models,
    server_model,
    clients_train_loaders,
    test_loader,
    num_rounds=10,
    local_epochs=1,
    max_clients_per_round=3,
    loss_fn=None,
    gamma_0=1e-3,
    alpha=0.1,
    delay_t=2,
    accumulation_steps=1,
    early_stopping_patience=10,
    mu=0.01
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    server_losses = []                            # proxy: mean of client task losses this round
    selected_clients_per_round = []               # list[list[int]]
    execution_times_by_round = []                 # list[list[float]]
    round_durations = []                          # list[float]
    comm_stats = []                               # per round comm

    test_losses_per_round = []
    accuracy_per_round = []

    # Static per-round model bytes (upload/download)
    model_bytes = state_dict_nbytes(server_model.state_dict())

    for round_num in tqdm(range(num_rounds), desc="FedProx Rounds"):
        round_start = time.time()

        selected = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected)
        exec_times = []

        # snapshot server weights at round start (for prox term)
        server_state_snapshot = {k: v.detach().clone() for k, v in server_model.state_dict().items()}

        async def train_client_task(i):
            t0 = time.time()
            state_dict, client_losses = await train_client_fedprox(
                clients_models[i],
                clients_train_loaders[i],
                device,
                local_epochs,
                loss_fn,
                gamma_0,
                alpha,
                delay_t=delay_t,
                accumulation_steps=accumulation_steps,
                early_stopping_patience=early_stopping_patience,
                server_state_at_round=server_state_snapshot,
                mu=mu
            )
            exec_time = time.time() - t0
            return state_dict, client_losses, exec_time

        results = await asyncio.gather(*[train_client_task(i) for i in selected])

        # Aggregate proxy loss
        round_loss = 0.0
        client_states = []
        for (_, (state_dict, client_loss, exec_time)) in enumerate(results):
            client_states.append(state_dict)
            round_loss += (sum(client_loss) / max(1, len(client_loss)))
            exec_times.append(exec_time)
        server_losses.append(round_loss / max(1, len(results)))
        execution_times_by_round.append(exec_times)

        # FedAvg aggregation (sample-size weighted)
        total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected)
        new_server_state_dict = {k: torch.zeros_like(v) for k, v in client_states[0].items()}
        for key in new_server_state_dict:
            for i, client_weight in zip(selected, client_states):
                weight_factor = len(clients_train_loaders[i].dataset) / max(1, total_samples)
                new_server_state_dict[key] += client_weight[key] * weight_factor
        server_model.load_state_dict(new_server_state_dict)

        # Evaluate on test set
        test_loss, test_acc = evaluate(server_model, test_loader, device, loss_fn)
        test_losses_per_round.append(test_loss)
        accuracy_per_round.append(test_acc)

        # Communication overhead (down+up) & round timing
        bytes_down = model_bytes * len(selected)  # broadcast server model to selected clients
        bytes_up   = model_bytes * len(selected)  # each selected client uploads updated weights once
        bytes_total = bytes_down + bytes_up
        comm_stats.append({
            "round": round_num,
            "num_clients": len(selected),
            "bytes_down": bytes_down,
            "bytes_up": bytes_up,
            "bytes_total": bytes_total,
            "mb_total": bytes_total / (1024.0 * 1024.0),
        })

        round_durations.append(time.time() - round_start)

    return (server_model, server_losses, selected_clients_per_round,
            execution_times_by_round, round_durations, test_losses_per_round, accuracy_per_round, comm_stats)

# -------------------------
# Main
# -------------------------
def main():
    torch.backends.cudnn.benchmark = True

    # Dataset
    train_dataset, test_dataset, in_ch, n_classes = get_dataset_and_meta(DATASET)

    # Partition non-IID
    client_data_indices = partition_non_iid(train_dataset, num_clients, n_classes, alpha=alpha_dirichlet)

    # Loaders
    def make_loader(ds, idxs=None, shuffle=True):
        if idxs is not None:
            ds = Subset(ds, idxs)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loaders = [make_loader(train_dataset, indices, shuffle=True) for indices in client_data_indices]
    test_loader = make_loader(test_dataset, None, shuffle=False)

    # Models
    clients_models = [SimpleCNN(input_channels=in_ch, num_classes=n_classes, hidden_channels=32, num_layers=3)
                      for _ in range(num_clients)]
    server_model = SimpleCNN(input_channels=in_ch, num_classes=n_classes, hidden_channels=32, num_layers=3)

    # Loss
    loss_fn = F.nll_loss

    # Results dir
    results_dir = create_directory(num_clients=num_clients,
                                   num_rounds=num_rounds,
                                   local_epochs=local_epochs,
                                   max_clients_per_round=num_clients_per_round,
                                   dataset_name=DATASET,
                                   base_dir=base_results_dir)

    # Run async FedProx
    (server_model, server_losses, selected_clients_per_round,
     execution_times_by_round, round_durations, test_losses_per_round,
     accuracy_per_round, comm_stats) = asyncio.run(
        federated_learning_fedprox(
            clients_models=clients_models,
            server_model=server_model,
            clients_train_loaders=train_loaders,
            test_loader=test_loader,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            max_clients_per_round=num_clients_per_round,
            loss_fn=loss_fn,
            gamma_0=gamma_0,
            alpha=alpha_staleness,
            delay_t=delay_t,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            mu=mu_prox
        )
    )

    # Save arrays + plots
    np.save(os.path.join(results_dir, "accuracy_per_round.npy"), np.array(accuracy_per_round))
    np.save(os.path.join(results_dir, "test_loss_per_round.npy"), np.array(test_losses_per_round))
    np.save(os.path.join(results_dir, "server_losses.npy"), np.array(server_losses))

    def plot_series_local(vals, title, xlab, ylab, fname):
        plot_series(vals, title=title, xlabel=xlab, ylabel=ylab, save_path=os.path.join(results_dir, fname))

    plot_series_local(accuracy_per_round, "FedProx — Test Accuracy per Round", "Round", "Accuracy", "accuracy_plot.png")
    plot_series_local(test_losses_per_round, "FedProx — Test Loss per Round", "Round", "Loss", "test_loss_plot.png")
    plot_series_local(server_losses, "FedProx — Server Proxy Loss Across Rounds", "Round", "Loss", "server_training_loss.png")

    # Selected clients & scatter
    save_selected_clients(selected_clients_per_round, os.path.join(results_dir, "selected_clients.csv"))
    plt.figure(figsize=(10, 6))
    for r, clients in enumerate(selected_clients_per_round):
        plt.scatter([r] * len(clients), clients, marker="x")
    plt.xlabel("Round")
    plt.ylabel("Client Index")
    plt.title("FedProx — Clients Selected Per Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "selected_clients_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Exec times, round times, comm
    save_execution_times(execution_times_by_round, os.path.join(results_dir, "execution_times.csv"))
    save_round_times(round_durations, os.path.join(results_dir, "round_times.csv"))
    save_comm_overhead(comm_stats, os.path.join(results_dir, "communication_overhead.csv"))

    # Summary
    summary = {
        "method": "async_fedprox",
        "dataset": DATASET,
        "num_clients": num_clients,
        "num_clients_per_round": num_clients_per_round,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "non_iid_alpha": alpha_dirichlet,
        "delay_t": delay_t,
        "gamma_0": gamma_0,
        "alpha_staleness": alpha_staleness,
        "mu_prox": mu_prox,
        "accuracy_final": float(accuracy_per_round[-1]) if accuracy_per_round else None,
        "best_accuracy": float(max(accuracy_per_round)) if accuracy_per_round else None,
        "total_runtime_sec": float(sum(round_durations)),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] FedProx results saved in: {results_dir}")

if __name__ == "__main__":
    main()
