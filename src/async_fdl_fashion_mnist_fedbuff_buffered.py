#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:51:56 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous Federated Learning — Buffered (FedBuff-style) Aggregator
- Event-driven: clients run continuously and send updates when ready.
- Server aggregates when a buffer of size K fills (or on timeout).
- Uses delta aggregation with sample-size weighting and staleness decay.

Artifacts saved (per aggregation event “agg_step”):
- Test accuracy/loss (accuracy_per_agg.npy, test_loss_per_agg.npy + plots)
- Server proxy loss from aggregated client updates (server_losses.npy + plot)
- Selected clients per aggregation (selected_clients.csv + scatter plot)
- Per-aggregation exec times (execution_times.csv), aggregation durations (agg_times.csv)
- Communication overhead per aggregation (communication_overhead.csv)
- Summary (summary.json)

Default dataset: Fashion-MNIST
"""

import os
import time
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import asyncio
import nest_asyncio
from tqdm import tqdm

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
DATASET = "fashion_mnist"   # options: "fashion_mnist", "svhn", "emnist_byclass"

num_clients = 10
alpha_dirichlet = 0.5       # non-IID control (lower = more skewed)
batch_size = 64

# FedBuff-style settings
buffer_size_K = 8            # aggregate when K updates arrive
max_aggregations = 1000      # total aggregation steps (like “rounds” but event-driven)
aggregate_timeout_s = 9999   # optional timeout (seconds). Large => mostly buffer-triggered

# Client local training
local_epochs = 5
gamma_0 = 1e-3
accumulation_steps = 1
early_stopping_patience = 10

# Staleness simulation & decay
max_delay_seconds = 2.0      # network/availability delay simulation
staleness_beta = 0.05        # decay = 1 / (1 + beta * staleness)

# Concurrency cap (avoid GPU/CPU overload)
max_parallel_clients = 5

base_results_dir = "results_fedbuff"
random_seed = 1337

# Allow nested event loops for asyncio
nest_asyncio.apply()
torch.backends.cudnn.benchmark = True
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# -----------------------------------------------------------
# Utilities (IO/plots)
# -----------------------------------------------------------
def create_directory(tag, base_dir="results_fedbuff"):
    dir_name = os.path.join(
        base_dir,
        f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_losses_as_numpy(losses, filename):
    np.save(filename, np.array(losses))

def save_selected_clients(selected_clients, filename):
    with open(filename, "w") as f:
        f.write("AggStep,SelectedClients\n")
        for step, clients in enumerate(selected_clients):
            f.write(f"{step},{' '.join(map(str, clients))}\n")

def save_execution_times(exec_times_by_step, filename):
    with open(filename, "w") as f:
        f.write("AggStep,ClientIndex,ExecutionTimeSeconds\n")
        for step, times in enumerate(exec_times_by_step):
            for cid, t in times:
                f.write(f"{step},{cid},{t:.6f}\n")

def save_agg_times(agg_durations, filename):
    with open(filename, "w") as f:
        f.write("AggStep,DurationSeconds\n")
        for step, dur in enumerate(agg_durations):
            f.write(f"{step},{dur:.6f}\n")

def save_comm_overhead(comm_rows, filename):
    with open(filename, "w") as f:
        f.write("AggStep,NumUpdates,BytesDown,BytesUp,BytesTotal,MBTotal\n")
        for row in comm_rows:
            f.write(f"{row['agg_step']},{row['num_updates']},{row['bytes_down']},{row['bytes_up']},"
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

def clone_state_dict(sd):
    return {k: v.detach().clone() for k, v in sd.items()}

def state_dict_delta(new_sd, base_sd):
    return {k: (new_sd[k] - base_sd[k]) for k in base_sd.keys()}

def add_scaled_(target_sd, src_sd, scale):
    for k in target_sd.keys():
        target_sd[k] += src_sd[k] * scale

# -----------------------------------------------------------
# Model
# -----------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x):
        return self.relu(self.conv(x))

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for 1x28x28 (MNIST/Fashion-MNIST/EMNIST) and 3x32x32 (SVHN).
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

# -----------------------------------------------------------
# Dataset factory
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Non-IID partition (Dirichlet)
# -----------------------------------------------------------
def partition_non_iid(dataset, num_clients, num_classes, alpha=0.5):
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
        leftover = cls_indices[start:]
        for idx_left in leftover:
            client_indices[np.random.randint(num_clients)].append(idx_left)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices

# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn):
    model.eval().to(device)
    total = 0
    correct = 0
    loss_sum = 0.0
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

# -----------------------------------------------------------
# Client coroutine (continuous)
# -----------------------------------------------------------
async def client_worker(
    cid,
    base_model_fn,
    train_loader,
    update_queue: asyncio.Queue,
    get_global_snapshot,        # callable -> (state_dict, version)
    device,
    gamma_0,
    accumulation_steps,
    early_stopping_patience,
    max_delay_seconds,
    stop_event: asyncio.Event,
    sem: asyncio.Semaphore,
    loss_fn
):
    """
    Each client repeatedly:
      - Sleeps random delay
      - Gets global snapshot (weights, version)
      - Trains locally (E epochs)
      - Sends (delta, num_samples, base_version, mean_loss, exec_time) to the server
    """
    model = base_model_fn().to(device)
    num_samples = len(train_loader.dataset)

    while not stop_event.is_set():
        # Simulate availability / network delay
        await asyncio.sleep(random.uniform(0, max_delay_seconds))

        # Snapshot of global model & version
        global_sd, base_version = get_global_snapshot()

        # Copy snapshot into local model
        model.load_state_dict(global_sd)

        patience, best_loss = 0, float("inf")
        step0 = time.time()

        # Limit concurrent trainers
        async with sem:
            for epoch in range(local_epochs):
                model.train()
                epoch_loss = 0.0
                # staleness-aware LR dampening (mild)
                gamma_t = gamma_0 / np.sqrt(epoch + 1)

                optimizer = torch.optim.Adam(model.parameters(), lr=float(gamma_t))
                for bidx, (x, y) in enumerate(train_loader):
                    x = x.to(device)
                    y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
                    y = y.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = loss_fn(out, y)
                    loss.backward()
                    if (bidx + 1) % accumulation_steps == 0:
                        optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / max(1, len(train_loader))
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        exec_time = time.time() - step0

        # Compute delta w.r.t. base snapshot
        with torch.no_grad():
            new_sd = clone_state_dict(model.state_dict())
            delta_sd = state_dict_delta(new_sd, global_sd)

        # Enqueue the update
        await update_queue.put({
            "cid": cid,
            "num_samples": num_samples,
            "base_version": base_version,
            "delta": delta_sd,                # state-dict delta
            "mean_loss": float(best_loss),
            "exec_time": float(exec_time),
        })

# -----------------------------------------------------------
# Server / Aggregator (event-driven) — with tqdm progress bar
# -----------------------------------------------------------
async def fedbuff_training_loop(
    base_model_fn,
    clients_loaders,
    test_loader,
    buffer_size_K,
    max_aggregations,
    aggregate_timeout_s,
    staleness_beta,
    device,
    loss_fn
):
    """
    Event-driven training loop:
      - start client workers
      - collect updates in a buffer
      - when buffer reaches K (or timeout), aggregate deltas with weights:
            w_i = num_samples_i * 1/(1 + beta * staleness_i)
            theta <- theta + (sum_i w_i * delta_i) / (sum_i w_i)
      - evaluate, log, and continue until max_aggregations
    """
    # Global model state & version counter
    global_model = base_model_fn().to(device)
    global_state = clone_state_dict(global_model.state_dict())
    global_version = 0

    def get_global_snapshot():
        # copy a snapshot (we clone tensors to keep isolation)
        return clone_state_dict(global_state), global_version

    # Queue for client updates
    update_queue = asyncio.Queue()

    # Concurrency limit for client training
    sem = asyncio.Semaphore(max_parallel_clients)

    # Stop event for clients
    stop_event = asyncio.Event()

    # Launch client tasks
    client_tasks = []
    for cid, loader in enumerate(clients_loaders):
        task = asyncio.create_task(
            client_worker(
                cid=cid,
                base_model_fn=base_model_fn,
                train_loader=loader,
                update_queue=update_queue,
                get_global_snapshot=get_global_snapshot,
                device=device,
                gamma_0=gamma_0,
                accumulation_steps=accumulation_steps,
                early_stopping_patience=early_stopping_patience,
                max_delay_seconds=max_delay_seconds,
                stop_event=stop_event,
                sem=sem,
                loss_fn=loss_fn
            )
        )
        client_tasks.append(task)

    # Logging
    accuracy_per_agg = []
    test_loss_per_agg = []
    server_proxy_loss = []
    selected_clients_per_agg = []
    exec_times_by_agg = []
    agg_durations = []
    comm_stats = []

    model_bytes = state_dict_nbytes(global_state)

    # Aggregation buffer
    buffer = []
    agg_count = 0
    last_agg_time = time.time()

    # tqdm progress bar
    pbar = tqdm(total=max_aggregations, desc="Aggregations", unit="agg")

    while agg_count < max_aggregations:
        # Wait for either: K updates, or timeout
        try:
            update = await asyncio.wait_for(update_queue.get(), timeout=aggregate_timeout_s)
            buffer.append(update)

            # Drain without blocking until K
            while len(buffer) < buffer_size_K:
                try:
                    update = update_queue.get_nowait()
                    buffer.append(update)
                except asyncio.QueueEmpty:
                    break
        except asyncio.TimeoutError:
            # timeout => aggregate whatever we have (if non-empty)
            pass

        if len(buffer) == 0:
            # nothing to aggregate yet, continue waiting
            # Show heartbeat in the bar so you know it's alive
            pbar.set_postfix_str("waiting...")
            continue

        # Aggregate now
        with torch.no_grad():
            current_version = global_version

            # Weighted sum of deltas
            weighted_delta = {k: torch.zeros_like(v) for k, v in global_state.items()}
            weights_sum = 0.0

            proxy_loss = 0.0
            cids = []
            execs = []
            num_updates_this_agg = len(buffer)

            for upd in buffer:
                cid = upd["cid"]
                base_version = upd["base_version"]
                staleness = max(0, current_version - base_version)
                decay = 1.0 / (1.0 + staleness_beta * float(staleness))
                w = upd["num_samples"] * decay

                add_scaled_(weighted_delta, upd["delta"], w)
                weights_sum += w

                proxy_loss += upd["mean_loss"]
                cids.append(cid)
                execs.append((cid, upd["exec_time"]))

            proxy_loss = proxy_loss / max(1, num_updates_this_agg)

            # theta <- theta + (sum_i w_i * delta_i) / sum_i w_i
            for k in global_state.keys():
                global_state[k] = global_state[k] + (weighted_delta[k] / max(1e-12, weights_sum))

            # Update model weights and version
            global_model.load_state_dict(global_state)
            global_version += 1

        # Evaluate
        tloss, tacc = evaluate(global_model, test_loader, device, loss_fn)

        # Log
        accuracy_per_agg.append(tacc)
        test_loss_per_agg.append(tloss)
        server_proxy_loss.append(proxy_loss)
        selected_clients_per_agg.append(cids)
        exec_times_by_agg.append(execs)

        # Comm overhead (down + up) for this aggregation — count only consumed updates
        bytes_down = model_bytes * num_updates_this_agg  # snapshot pulled
        bytes_up   = model_bytes * num_updates_this_agg  # update sent
        comm_stats.append({
            "agg_step": agg_count,
            "num_updates": num_updates_this_agg,
            "bytes_down": bytes_down,
            "bytes_up": bytes_up,
            "bytes_total": bytes_down + bytes_up,
            "mb_total": (bytes_down + bytes_up) / (1024.0 * 1024.0),
        })

        # timing
        agg_durations.append(time.time() - last_agg_time)
        last_agg_time = time.time()

        # Update progress bar with live metrics
        pbar.set_postfix({
            "Acc": f"{tacc:.3f}",
            "Loss": f"{tloss:.3f}",
            "Upd": num_updates_this_agg
        })
        pbar.update(1)

        # Prepare next
        buffer.clear()
        agg_count += 1

    # Stop clients
    stop_event.set()
    for t in client_tasks:
        t.cancel()
        try:
            await t
        except:
            pass

    pbar.close()

    return {
        "accuracy_per_agg": accuracy_per_agg,
        "test_loss_per_agg": test_loss_per_agg,
        "server_proxy_loss": server_proxy_loss,
        "selected_clients_per_agg": selected_clients_per_agg,
        "exec_times_by_agg": exec_times_by_agg,
        "agg_durations": agg_durations,
        "comm_stats": comm_stats,
        "final_model": global_model,
    }

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset, test_dataset, in_ch, n_classes = get_dataset_and_meta(DATASET)

    # Partition non-IID
    client_indices = partition_non_iid(train_dataset, num_clients, n_classes, alpha=alpha_dirichlet)

    # Loaders
    def make_loader(ds, idxs=None, shuffle=True):
        if idxs is not None:
            ds = Subset(ds, idxs)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loaders = [make_loader(train_dataset, idxs, shuffle=True) for idxs in client_indices]
    test_loader = make_loader(test_dataset, None, shuffle=False)

    # Model factory (fresh instance per client / server)
    def base_model_fn():
        return SimpleCNN(input_channels=in_ch, num_classes=n_classes, hidden_channels=32, num_layers=3)

    loss_fn = F.nll_loss

    tag = (f"{DATASET}_fedbuffK{buffer_size_K}_C{num_clients}"
           f"_E{local_epochs}_A{max_aggregations}_alpha{alpha_dirichlet}")
    results_dir = create_directory(tag, base_dir=base_results_dir)

    # Run FedBuff-style training
    results = asyncio.run(
        fedbuff_training_loop(
            base_model_fn=base_model_fn,
            clients_loaders=train_loaders,
            test_loader=test_loader,
            buffer_size_K=buffer_size_K,
            max_aggregations=max_aggregations,
            aggregate_timeout_s=aggregate_timeout_s,
            staleness_beta=staleness_beta,
            device=device,
            loss_fn=loss_fn
        )
    )

    # Save artifacts
    np.save(os.path.join(results_dir, "accuracy_per_agg.npy"), np.array(results["accuracy_per_agg"]))
    np.save(os.path.join(results_dir, "test_loss_per_agg.npy"), np.array(results["test_loss_per_agg"]))
    save_losses_as_numpy(results["server_proxy_loss"], os.path.join(results_dir, "server_losses.npy"))

    plot_series(results["accuracy_per_agg"], "Test Accuracy per Aggregation",
                "Aggregation step", "Accuracy", os.path.join(results_dir, "accuracy_plot.png"))
    plot_series(results["test_loss_per_agg"], "Test Loss per Aggregation",
                "Aggregation step", "Loss", os.path.join(results_dir, "test_loss_plot.png"))
    plot_series(results["server_proxy_loss"], "Server Proxy Loss per Aggregation",
                "Aggregation step", "Loss", os.path.join(results_dir, "server_training_loss.png"))

    # Selected clients scatter
    save_selected_clients(results["selected_clients_per_agg"], os.path.join(results_dir, "selected_clients.csv"))
    plt.figure(figsize=(10, 6))
    for step, cids in enumerate(results["selected_clients_per_agg"]):
        plt.scatter([step]*len(cids), cids, marker="x")
    plt.xlabel("Aggregation step")
    plt.ylabel("Client Index")
    plt.title("Clients Contributing per Aggregation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "selected_clients_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Exec times
    save_execution_times(results["exec_times_by_agg"], os.path.join(results_dir, "execution_times.csv"))

    # Agg durations
    save_agg_times(results["agg_durations"], os.path.join(results_dir, "agg_times.csv"))

    # Communication overhead
    save_comm_overhead(results["comm_stats"], os.path.join(results_dir, "communication_overhead.csv"))

    # Summary
    summary = {
        "dataset": DATASET,
        "num_clients": num_clients,
        "buffer_size_K": buffer_size_K,
        "max_aggregations": max_aggregations,
        "local_epochs": local_epochs,
        "non_iid_alpha": alpha_dirichlet,
        "staleness_beta": staleness_beta,
        "max_delay_seconds": max_delay_seconds,
        "best_accuracy": max(results["accuracy_per_agg"]) if results["accuracy_per_agg"] else None,
        "final_accuracy": results["accuracy_per_agg"][-1] if results["accuracy_per_agg"] else None,
        "total_runtime_sec": float(sum(results["agg_durations"])),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] FedBuff-style results saved in: {results_dir}")

if __name__ == "__main__":
    main()
