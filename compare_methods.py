"""
Main script to compare federated learning methods on MNIST dataset.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.mnist_net import MNISTNet
from src.data.data_loader import load_mnist, create_federated_data, get_data_loaders
from src.algorithms.fedavg import FedAvg
from src.algorithms.fedprox import FedProx


def plot_comparison(results, save_path='results_comparison.png'):
    """Plot comparison of different methods."""
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for method_name, history in results.items():
        plt.plot(history['accuracy'], label=method_name, marker='o', markersize=4)
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for method_name, history in results.items():
        plt.plot(history['loss'], label=method_name, marker='s', markersize=4)
    plt.xlabel('Communication Round')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def main():
    # Configuration
    num_clients = 10
    num_rounds = 20
    local_epochs = 5
    learning_rate = 0.01
    batch_size = 32
    client_fraction = 1.0  # Fraction of clients to use per round
    iid = True  # IID or non-IID data distribution
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Federated Learning Methods Comparison on MNIST")
    print("=" * 60)
    print(f"Number of clients: {num_clients}")
    print(f"Communication rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Client fraction: {client_fraction}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()
    
    print(f"Creating federated data for {num_clients} clients...")
    client_datasets = create_federated_data(train_dataset, num_clients, iid=iid)
    client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, batch_size)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Average samples per client: {len(train_dataset) // num_clients}")
    
    # Dictionary to store results
    results = {}
    
    # Train FedAvg
    print("\n" + "=" * 60)
    print("Training with FedAvg")
    print("=" * 60)
    model_fedavg = MNISTNet()
    fedavg = FedAvg(
        model_fedavg,
        client_loaders,
        test_loader,
        device,
        local_epochs=local_epochs,
        learning_rate=learning_rate
    )
    results['FedAvg'] = fedavg.train(num_rounds, client_fraction)
    
    # Train FedProx
    print("\n" + "=" * 60)
    print("Training with FedProx (mu=0.01)")
    print("=" * 60)
    model_fedprox = MNISTNet()
    fedprox = FedProx(
        model_fedprox,
        client_loaders,
        test_loader,
        device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        mu=0.01
    )
    results['FedProx (mu=0.01)'] = fedprox.train(num_rounds, client_fraction)
    
    # Train FedProx with different mu
    print("\n" + "=" * 60)
    print("Training with FedProx (mu=0.1)")
    print("=" * 60)
    model_fedprox2 = MNISTNet()
    fedprox2 = FedProx(
        model_fedprox2,
        client_loaders,
        test_loader,
        device,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        mu=0.1
    )
    results['FedProx (mu=0.1)'] = fedprox2.train(num_rounds, client_fraction)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    for method_name, history in results.items():
        final_acc = history['accuracy'][-1]
        final_loss = history['loss'][-1]
        max_acc = max(history['accuracy'])
        print(f"{method_name:25} - Final: {final_acc:.2f}%, Max: {max_acc:.2f}%, Loss: {final_loss:.4f}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
