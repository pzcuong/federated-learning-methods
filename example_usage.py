"""
Example script showing how to use individual federated learning algorithms.
"""
import torch
import numpy as np
from src.models.mnist_net import MNISTNet
from src.data.data_loader import load_mnist, create_federated_data, get_data_loaders
from src.algorithms.fedavg import FedAvg


def run_fedavg_example():
    """Example of using FedAvg algorithm."""
    # Configuration
    num_clients = 5
    num_rounds = 10
    local_epochs = 3
    learning_rate = 0.01
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("FedAvg Example")
    print("=" * 60)
    
    # Load data
    print("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()
    
    # Create federated data
    print(f"Creating federated data for {num_clients} clients...")
    client_datasets = create_federated_data(train_dataset, num_clients, iid=True)
    client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, batch_size)
    
    # Create model
    model = MNISTNet()
    
    # Initialize FedAvg
    fedavg = FedAvg(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    
    # Train
    print(f"\nTraining for {num_rounds} rounds...")
    history = fedavg.train(num_rounds=num_rounds, client_fraction=1.0)
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final Test Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Final Test Loss: {history['loss'][-1]:.4f}")
    print(f"Best Test Accuracy: {max(history['accuracy']):.2f}%")


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_fedavg_example()
