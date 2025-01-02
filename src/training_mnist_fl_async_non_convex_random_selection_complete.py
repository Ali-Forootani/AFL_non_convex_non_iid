#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Code for Federated Learning with MNIST
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nest_asyncio
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mnist_deep_framework import CNNMnistModel
from mnist_data_preparation import MNISTDataPreparationFL
from mnist_trainer import federated_learning_with_mnist_cnn_4
import asyncio

# Allow nested event loops for asyncio
nest_asyncio.apply()

# Utility to set directory
def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

# Set root directory
root_dir = setting_directory(2)

# Fix seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(7)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Reduce training dataset size for quick prototyping
reduced_train_size = 60000
train_subset = Subset(train_dataset, range(reduced_train_size))

# Federated Learning Setup
num_clients = 10
batch_size = 32
num_rounds = 100
local_epochs = 100
aggregation_method = "weighted_average"
accumulation_steps = 50
early_stopping_patience = 100
max_clients_per_round = 5
num_layers = 4

# Prepare data for federated learning
mnist_data_preparation = MNISTDataPreparationFL(train_subset, test_dataset, num_clients, batch_size)
client_indices = mnist_data_preparation.partition_data()
train_loaders, test_loaders = mnist_data_preparation.get_client_loaders(client_indices)

# Initialize clients' models, optimizers, and schedulers
clients_models = []
clients_optimizers = []
clients_schedulers = []

for i in range(num_clients):
    cnn_model = CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=64, num_layers=num_layers)
    optimizer = cnn_model.optimizer_func()
    scheduler = cnn_model.scheduler_setting()

    clients_models.append(cnn_model.to(device))
    clients_optimizers.append(optimizer)
    clients_schedulers.append(scheduler)

# Initialize global model
global_model = CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=64, num_layers=num_layers).to(device)

# Async main function
async def main():
    save_dir = "models/"
    os.makedirs(save_dir, exist_ok=True)

    # Federated learning parameters
    gamma_0 = 1e-3
    alpha = 0.01

    # Execute federated learning
    (
        server_model, training_losses,
        server_training_losses, selected_clients_by_round, execution_times_by_round
    ) = await federated_learning_with_mnist_cnn_4(
        clients_models=clients_models,
        server_model=global_model,
        clients_train_loaders=train_loaders,
        clients_test_loaders=test_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        min_clients_per_round=1,
        max_clients_per_round=max_clients_per_round,
        aggregation_method=aggregation_method,
        loss_fn=torch.nn.CrossEntropyLoss(),
        gamma_0=gamma_0,
        alpha=alpha,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir
    )

    # Replace None with np.nan for saving
    cleaned_training_losses = [
        [np.nan if val is None else val for val in round_losses]
        for round_losses in training_losses
    ]

    # Save results
    np.save(os.path.join(save_dir, "training_losses.npy"), cleaned_training_losses)
    np.save(os.path.join(save_dir, "server_training_losses.npy"), server_training_losses)
    np.save(os.path.join(save_dir, "selected_clients_by_round.npy"), selected_clients_by_round)
    np.save(os.path.join(save_dir, "execution_times_by_round.npy"), execution_times_by_round)

    # Log key results
    print(f"\nFederated learning completed. Results saved in {save_dir}")
    print(f"Server training losses over rounds: {server_training_losses}")
    print(f"Execution times by round: {execution_times_by_round}")
    print(f"Selected clients by round: {selected_clients_by_round}")

    # Save server model
    torch.save(server_model.state_dict(), os.path.join(save_dir, "server_model_final.pth"))

    # Plot results
    plot_losses(save_dir, num_clients=len(clients_models), num_rounds=num_rounds)
    plot_selected_clients(selected_clients_by_round, os.path.join(save_dir, "selected_clients_by_round.png"))
    plot_execution_times(execution_times_by_round, os.path.join(save_dir, "execution_times_by_round.png"))

# Function to plot losses
def plot_losses(save_dir, num_clients, num_rounds):
    training_losses = np.load(os.path.join(save_dir, "training_losses.npy"), allow_pickle=True)
    server_training_losses = np.load(os.path.join(save_dir, "server_training_losses.npy"), allow_pickle=True)

    # Plot client training losses
    plt.figure(figsize=(10, 6))
    for client_id, client_losses in enumerate(training_losses):
        plt.plot(client_losses, label=f"Client {client_id + 1}", linewidth=2)
    plt.title("Training Losses per Client Across Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.yscale('log')  # Logarithmic scale for y-axis
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/client_training_losses_plot.png")
    plt.show()

    # Plot server training losses
    plt.figure(figsize=(10, 6))
    plt.plot(server_training_losses, label="Server Model", linewidth=2, marker="o")
    plt.title("Server Model Training Loss Across Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.yscale('log')  # Logarithmic scale for y-axis
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/server_training_losses_plot.png")
    plt.show()

# Plot selected clients by round
def plot_selected_clients(selected_clients, save_path):
    plt.figure(figsize=(10, 6))
    for round_num, clients in enumerate(selected_clients):
        for client in clients:
            plt.scatter(round_num, client, label=f"Client {client}" if round_num == 0 else "", s=50)
    plt.title("Selected Clients Per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Client ID")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Plot execution times by round
def plot_execution_times(execution_times, save_path):
    mean_times = [np.mean(times) for times in execution_times]
    plt.figure(figsize=(10, 6))
    plt.plot(mean_times, marker="o", label="Mean Execution Time")
    plt.title("Mean Execution Time Per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Run the async main function
asyncio.run(main())
