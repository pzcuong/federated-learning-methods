# Federated Learning Methods

A research project to demonstrate and compare different federated learning algorithms on the MNIST dataset.

## Overview

This project implements and compares several federated learning methods:
- **FedAvg (Federated Averaging)**: The standard federated learning algorithm
- **FedProx (Federated Proximal)**: An extension that adds a proximal term to handle heterogeneous data

## Features

- Clean, modular implementation of federated learning algorithms
- Support for both IID and non-IID data distributions
- Comprehensive comparison and visualization
- Easy to extend with new algorithms

## Project Structure

```
federated-learning-methods/
├── src/
│   ├── algorithms/          # Federated learning algorithms
│   │   ├── fedavg.py       # Federated Averaging
│   │   └── fedprox.py      # Federated Proximal
│   ├── data/               # Data loading and preprocessing
│   │   └── data_loader.py  # MNIST data loader with federated splits
│   ├── models/             # Neural network models
│   │   └── mnist_net.py    # CNN model for MNIST
│   └── utils/              # Utility functions
│       └── training_utils.py  # Training and evaluation utilities
├── compare_methods.py      # Main script to compare methods
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pzcuong/federated-learning-methods.git
cd federated-learning-methods
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the comparison script to train and compare different federated learning methods:

```bash
python compare_methods.py
```

This will:
1. Download and prepare the MNIST dataset
2. Split data across federated clients
3. Train models using FedAvg and FedProx
4. Generate comparison plots
5. Display final results

### Configuration

You can modify the hyperparameters in `compare_methods.py`:

```python
num_clients = 10          # Number of federated clients
num_rounds = 20           # Number of communication rounds
local_epochs = 5          # Local training epochs per round
learning_rate = 0.01      # Learning rate
batch_size = 32           # Batch size
client_fraction = 1.0     # Fraction of clients to use per round
iid = True                # IID or non-IID data distribution
```

### Example Output

```
==============================================================
Federated Learning Methods Comparison on MNIST
==============================================================
Number of clients: 10
Communication rounds: 20
Local epochs: 5
Learning rate: 0.01
Batch size: 32
Client fraction: 1.0
Data distribution: IID
Device: cpu
==============================================================

Training with FedAvg
Round 1/20 - Test Accuracy: 92.15%, Test Loss: 0.2534
Round 2/20 - Test Accuracy: 94.23%, Test Loss: 0.1892
...

Training with FedProx (mu=0.01)
Round 1/20 - Test Accuracy: 91.87%, Test Loss: 0.2612
Round 2/20 - Test Accuracy: 94.45%, Test Loss: 0.1856
...

Final Results
FedAvg                    - Final: 98.12%, Max: 98.34%, Loss: 0.0542
FedProx (mu=0.01)         - Final: 98.23%, Max: 98.45%, Loss: 0.0521
FedProx (mu=0.1)          - Final: 97.89%, Max: 98.11%, Loss: 0.0589
```

## Algorithms

### FedAvg (Federated Averaging)

FedAvg is the baseline federated learning algorithm where:
1. A subset of clients download the global model
2. Each client trains locally on their data
3. Client models are aggregated using weighted averaging
4. The process repeats for multiple rounds

### FedProx (Federated Proximal)

FedProx extends FedAvg by adding a proximal term to the local objective:
- Helps handle systems and statistical heterogeneity
- The proximal term μ/2 ||w - w_t||² keeps local models close to the global model
- More robust to non-IID data and varying amounts of local computation

## Research Applications

This project can be used to:
- Study the behavior of federated learning algorithms
- Compare performance on IID vs non-IID data
- Experiment with different model architectures
- Test new federated learning algorithms
- Analyze convergence and communication efficiency

## Extending the Project

### Adding a New Algorithm

1. Create a new file in `src/algorithms/`
2. Implement a class with `train_round()` and `train()` methods
3. Add it to the comparison in `compare_methods.py`

### Using a Different Dataset

1. Add data loading logic in `src/data/data_loader.py`
2. Create or modify the model in `src/models/`
3. Update the comparison script accordingly

## References

- **FedAvg**: McMahan, H. B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.
- **FedProx**: Li, T., et al. "Federated optimization in heterogeneous networks." MLSys 2020.

## License

This project is for research and educational purposes.