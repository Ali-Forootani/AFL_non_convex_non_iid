
# Asynchronous Federated Learning with PyTorch

This repository implements **Asynchronous Federated Learning (AFL)** using pure PyTorch, focusing on handling non-convex optimization tasks. It leverages Python's `asyncio` library to simulate asynchronous communication between clients. The project uses the MNIST dataset for classification to demonstrate AFL's efficiency and robustness.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

Federated Learning (FL) allows decentralized training of machine learning models while preserving data privacy. Traditional FL faces challenges such as:
- **Communication Overhead**
- **Straggler Effects** from slow clients
- **System Heterogeneity**

Asynchronous Federated Learning (AFL) resolves these challenges by enabling clients to update the server independently of others. This implementation demonstrates AFL with:
- Non-convex objective functions
- Simulated delays for client updates
- Robust handling of client heterogeneity

---

## Features

- **Asynchronous Updates**: Clients communicate updates independently, without waiting for synchronization.
- **Scalable Architecture**: Easily adaptable for larger datasets and client pools.
- **Privacy Preserving**: Client data remains local and secure.
- **Pure PyTorch Implementation**: No external federated learning frameworks required.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the main script:
   ```bash
   python train_async_fl.py
   ```

2. Adjust training parameters in `config.py`:
   - Number of clients
   - Learning rate
   - Number of communication rounds
   - Simulated client delay configurations

---

## Implementation Details

### Architecture
- **Model**: A Convolutional Neural Network (CNN) for MNIST digit classification.
- **Client Simulation**: Clients train locally and send model updates asynchronously.
- **Server**: Aggregates client updates and updates the global model without waiting for all clients.

### Workflow
1. Initialize a global model on the server.
2. Simulate multiple clients with local datasets.
3. Clients perform local training and asynchronously send updates to the server.
4. The server aggregates updates and refines the global model.
5. Repeat for a predefined number of communication rounds.

---

## Results

- [Insert accuracy or performance metrics here]
- [Visualizations: Add graphs if applicable]

---

## Contributing

Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.







# Asynchronous Federated Learning with non-convex client objective functions and heterogeneous dataset

This repository implements **asynchronous federated learning** using **pure PyTorch**, training **CNN and ResNet models** on **MNIST and CIFAR-10** datasets. The implementation does **not** use external federated learning frameworks like Flower or PySyft, instead leveraging **asyncio** for asynchronous client updates.

## Features

- **General & Configurable Framework**:
  - The number of clients, local epochs, number of rounds, clients per round, batch size, and other parameters can be easily set as inputs.
  - Works with **any image classification dataset** that can be loaded using PyTorch's `torchvision.datasets`.
- **Asynchronous Federated Learning**: Clients train asynchronously with simulated network delays.
- **Support for MNIST & CIFAR-10**:
  - **MNIST**: CNN-based classifier.
  - **CIFAR-10**: ResNet-based classifier.
- **Non-IID Data Partitioning**: Uses **Dirichlet distribution** for imbalanced data distribution among clients.
- **Delay-Aware Learning Rate**: Adjusts learning rate based on client delays.
- **Early Stopping**: Clients stop training if loss does not improve for a given number of epochs.
- **Loss Tracking & Visualization**: Saves and plots client/server training loss over rounds.
- **Client Selection Tracking**: Saves and visualizes selected clients per round.
- **Execution Time Analysis**: Logs and saves execution times for each client per round.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- asyncio
- nest_asyncio

You can install dependencies using:

```bash
pip install torch torchvision numpy matplotlib tqdm nest_asyncio
```

## Running the Code

To start federated learning on **MNIST**:

```bash
python async_fdl_mnist_delay_track_lr_non_iid.py
```

To start federated learning on **CIFAR-10**:

```bash
python async_fdl_cifar_delay_track_lr_non_iid.py
```

### Configurable Parameters:

| Parameter | Description | Example |
|-----------|------------|---------|
| `--num_clients` | Number of participating clients | `10` |
| `--num_rounds` | Number of federated learning rounds | `1000` |
| `--local_epochs` | Local training epochs per client | `10` |
| `--clients_per_round` | Number of clients selected per round | `5` |
| `--batch_size` | Training batch size | `64` |
| `--alpha` | Dirichlet distribution parameter for non-IID partitioning | `0.5` |
| `--gamma_0` | Initial learning rate | `1e-3` |
| `--alpha_lr` | Adjustment factor for delay-aware learning rate | `0.01` |
| `--delay_t` | Maximum simulated network delay (seconds) | `2` |
| `--early_stopping` | Number of epochs before stopping training early | `10` |

This makes the implementation **flexible** and **adaptable** to different datasets, models, and training settings.

## Project Structure

```
.
├── data/                   # Directory for downloading datasets
├── results/                # Directory for saving training results
│   ├── mnist_clients_*/    # MNIST results
│   ├── cifar_clients_*/    # CIFAR-10 results
│   ├── client_X_losses.npy # Training loss per client
│   ├── server_losses.npy   # Training loss per round at server
│   ├── execution_times.csv # Execution time per client per round
│   ├── selected_clients.csv# Selected clients per round
│   ├── plots/              # Saved loss and client selection plots
│
├── mnist_main.py           # MNIST federated learning script
├── cifar_main.py           # CIFAR-10 federated learning script
├── model_mnist.py          # CNN Model for MNIST
├── model_cifar.py          # ResNet Model for CIFAR-10
├── utils.py                # Utility functions for saving/loading data
├── README.md               # This file
```

## Dataset & Model Details

### **MNIST**
- **Model:** CNN with multiple convolutional layers.
- **Default Settings**:
  - **Number of Clients:** 10
  - **Rounds:** 1000
  - **Local Epochs:** 10
  - **Clients per Round:** 5
  - **Batch Size:** 64

### **CIFAR-10**
- **Model:** ResNet with Basic Blocks.
- **Default Settings**:
  - **Number of Clients:** 10
  - **Rounds:** 200
  - **Local Epochs:** 10
  - **Clients per Round:** 6
  - **Batch Size:** 64

## Results & Visualization

Once training is complete, results will be saved in the `results/` directory.

- **Client Training Losses:** Saved as `.npy` files and plotted per client.
- **Server Loss Across Rounds:** Saved as `.npy` and plotted.
- **Selected Clients Per Round:** Logged in `selected_clients.csv` and visualized in `selected_clients_plot.png`.
- **Execution Times:** Logged in `execution_times.csv`.

### Example Training Loss Plot (MNIST & CIFAR-10):
![Server Training Loss](results/server_training_loss.png)

### Example Client Selection Plot:
![Selected Clients Per Round](results/selected_clients_plot.png)

## Extending the Code

Since the code is **general**, it can be extended to **other datasets and models**:
1. **Change the dataset**:
   - Replace `datasets.MNIST` or `datasets.CIFAR10` with any PyTorch-supported dataset (e.g., `datasets.FashionMNIST`, `datasets.CIFAR100`).
2. **Modify the model**:
   - Update `model_mnist.py` or `model_cifar.py` to define a new architecture (e.g., MobileNet, Transformer-based models).
3. **Tune hyperparameters**:
   - Adjust **learning rate, batch size, local epochs, client selection strategy**, etc.
4. **Implement secure aggregation**:
   - Add **differential privacy** or **secure aggregation** for enhanced security.

## Future Improvements

- Support for **heterogeneous models** across clients.
- Implement **secure aggregation** for privacy.
- Expand to **other datasets** beyond MNIST & CIFAR-10.
- Optimize **communication efficiency** for real-world deployment.

## Citation

If you use this repository in your research, please cite it:

```
@misc{yourgithubrepo,
  author = {Forootan},
  title = {Asynchronous Federated Learning with PyTorch (MNIST & CIFAR-10)},
  year = {2025},
  url = {https://github.com/yourusername/async-federated-learning}
}
```

---

This version clearly highlights that the code is **general** and **configurable** for different datasets, clients, and hyperparameters. Let me know if you need any more refinements! 🚀
