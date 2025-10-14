"""
Demo script with synthetic data to showcase the framework without downloading MNIST.
This is useful for testing the framework structure.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from src.models.mnist_net import MNISTNet
from src.algorithms.fedavg import FedAvg
from src.algorithms.fedprox import FedProx


def create_synthetic_mnist_data(num_samples=1000, num_test=200):
    """Create synthetic MNIST-like data for testing."""
    # Create random images (28x28, grayscale)
    train_images = torch.randn(num_samples, 1, 28, 28)
    # Create random labels (0-9)
    train_labels = torch.randint(0, 10, (num_samples,))
    
    test_images = torch.randn(num_test, 1, 28, 28)
    test_labels = torch.randint(0, 10, (num_test,))
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    return train_dataset, test_dataset


def split_data_federated(dataset, num_clients):
    """Split dataset among clients."""
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        indices = list(range(start_idx, end_idx))
        client_datasets.append(Subset(dataset, indices))
    
    return client_datasets


def main():
    print("=" * 60)
    print("Federated Learning Framework Demo (Synthetic Data)")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    num_rounds = 5
    local_epochs = 2
    learning_rate = 0.01
    batch_size = 32
    device = torch.device('cpu')
    
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    train_dataset, test_dataset = create_synthetic_mnist_data(
        num_samples=600, num_test=100
    )
    
    # Split among clients
    client_datasets = split_data_federated(train_dataset, num_clients)
    
    # Create data loaders
    client_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Samples per client: {len(train_dataset) // num_clients}")
    
    # Test FedAvg
    print("\n" + "=" * 60)
    print("Testing FedAvg Algorithm")
    print("=" * 60)
    model_fedavg = MNISTNet()
    fedavg = FedAvg(
        model=model_fedavg,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    
    fedavg_history = fedavg.train(num_rounds=num_rounds, client_fraction=1.0)
    
    # Test FedProx
    print("\n" + "=" * 60)
    print("Testing FedProx Algorithm")
    print("=" * 60)
    model_fedprox = MNISTNet()
    fedprox = FedProx(
        model=model_fedprox,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        mu=0.01
    )
    
    fedprox_history = fedprox.train(num_rounds=num_rounds, client_fraction=1.0)
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    print(f"FedAvg  - Final Accuracy: {fedavg_history['accuracy'][-1]:.2f}%")
    print(f"FedProx - Final Accuracy: {fedprox_history['accuracy'][-1]:.2f}%")
    print("\n✓ Demo completed successfully!")
    print("\nNote: This demo uses synthetic random data.")
    print("For real MNIST experiments, run: python compare_methods.py")


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
