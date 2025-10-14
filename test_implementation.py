"""
Quick test script to verify the implementation works.
"""
import torch
import numpy as np
from src.models.mnist_net import MNISTNet
from src.data.data_loader import load_mnist, create_federated_data, get_data_loaders
from src.algorithms.fedavg import FedAvg

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("Loading MNIST dataset...")
train_dataset, test_dataset = load_mnist()
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

print("\nCreating federated data for 3 clients...")
client_datasets = create_federated_data(train_dataset, num_clients=3, iid=True)
client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, batch_size=64)

print("Creating model...")
model = MNISTNet()
device = torch.device('cpu')

print("\nInitializing FedAvg...")
fedavg = FedAvg(
    model=model,
    client_loaders=client_loaders,
    test_loader=test_loader,
    device=device,
    local_epochs=1,
    learning_rate=0.01
)

print("Running 2 training rounds...")
history = fedavg.train(num_rounds=2, client_fraction=1.0)

print("\n" + "=" * 60)
print("Test completed successfully!")
print(f"Round 1 - Accuracy: {history['accuracy'][0]:.2f}%, Loss: {history['loss'][0]:.4f}")
print(f"Round 2 - Accuracy: {history['accuracy'][1]:.2f}%, Loss: {history['loss'][1]:.4f}")
print("=" * 60)
