#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:29:21 2025

@author: forootan
"""

import numpy as np
import matplotlib.pyplot as plt
import os



import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


root_dir = setting_directory(0)






def load_and_plot_saved_losses(results_dir):
    
    """
    Load saved NumPy arrays and plot the training losses.
    """
    
    client_losses_files = [f for f in os.listdir(results_dir) if f.startswith("client_") and f.endswith("_losses.npy")]
    server_losses_file = os.path.join(results_dir, "server_losses.npy")

    # Load and plot client losses
    plt.figure(figsize=(10, 6))
    for client_file in client_losses_files:
        client_index = int(client_file.split("_")[1])
        client_losses = np.load(os.path.join(results_dir, client_file))
        plt.plot(client_losses, label=f"Client {client_index}")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.yscale("log")
    plt.xscale("log")
    plt.title("Client Training Losses")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Load and plot server losses
    if os.path.exists(server_losses_file):
        server_losses = np.load(server_losses_file)
        plt.figure(figsize=(10, 6))
        plt.plot(server_losses, linewidth= 1, label="Asynchronous Federated Learning", marker='o' )
        plt.xlabel("Rounds", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        #plt.yscale("log")
        plt.xscale("log")
        plt.title("Server Loss Across Rounds in CIFAR dataset with 50% clients participation", fontsize=18)
        plt.legend()
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)  # Set x-tick font size
        plt.yticks(fontsize=16)  # Set y-tick font size
        plt.grid(True)
        

        plt.savefig(root_dir + "/results/" +"asynch_nonconvex_mnist_server_losses_clients_4.png", dpi=300, bbox_inches='tight')  # High-quality PNG
        plt.savefig(root_dir + "/results/"+ "asynch_nonconvex_mnist_server_losses_clients_4.pdf", bbox_inches='tight')  # PDF format
        plt.show()
    else:
        print("Server losses file not found.")


# Example usage
results_dir = root_dir + "/results/mnist_clients_10_rounds_1000_epochs_10_clients_per_round_4_20250218_103407"  # Replace with your actual results directory
load_and_plot_saved_losses(results_dir)
