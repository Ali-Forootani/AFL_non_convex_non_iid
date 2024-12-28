
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

