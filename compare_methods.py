"""
Comprehensive comparison of federated learning methods on MNIST and CIFAR10.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.unified_cnn import UnifiedCNN
from src.data.data_loader import load_dataset, create_federated_data, get_data_loaders
from src.algorithms.fedavg import FedAvg
from src.algorithms.fedprox import FedProx
from src.algorithms.fed_nolowe import FedNoLowe
from src.algorithms.fed_lws import FedLWS
from src.algorithms.fed_awa import FedAWA


def plot_comparison(results, dataset_name, save_path=None):
    """Plot comparison of different methods."""
    if save_path is None:
        save_path = f'{dataset_name}_comparison.png'
    
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for method_name, history in results.items():
        plt.plot(history['accuracy'], label=method_name, marker='o', markersize=4)
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy Comparison ({dataset_name.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for method_name, history in results.items():
        plt.plot(history['loss'], label=method_name, marker='s', markersize=4)
    plt.xlabel('Communication Round')
    plt.ylabel('Test Loss')
    plt.title(f'Test Loss Comparison ({dataset_name.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def run_experiment(dataset_name, num_clients=10, num_rounds=20, local_epochs=5, 
                   learning_rate=0.01, batch_size=32, client_fraction=1.0, iid=True):
    """
    Run federated learning experiment on a specified dataset.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
        num_clients: Number of federated clients
        num_rounds: Number of communication rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        batch_size: Batch size
        client_fraction: Fraction of clients to use per round
        iid: IID or non-IID data distribution
    """
    # Use MPS (Metal) on Mac, CUDA on NVIDIA GPUs, or CPU as fallback
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("\n" + "=" * 70)
    print(f"Federated Learning Comparison on {dataset_name.upper()}")
    print("=" * 70)
    print(f"Number of clients: {num_clients}")
    print(f"Communication rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Client fraction: {client_fraction}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Load and prepare data
    print(f"\nLoading {dataset_name.upper()} dataset...")
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    print(f"Creating federated data for {num_clients} clients...")
    client_datasets = create_federated_data(train_dataset, num_clients, iid=iid)
    client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, batch_size)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Average samples per client: {len(train_dataset) // num_clients}")
    
    # Dictionary to store results
    results = {}
    
    # 1. FedAvg
    print("\n" + "=" * 70)
    print("Training with FedAvg")
    print("=" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fedavg = FedAvg(
        model, client_loaders, test_loader, device,
        local_epochs=local_epochs, learning_rate=learning_rate
    )
    results['FedAvg'] = fedavg.train(num_rounds, client_fraction)
    
    # 2. FedProx
    print("\n" + "=" * 70)
    print("Training with FedProx (mu=0.01)")
    print("=" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fedprox = FedProx(
        model, client_loaders, test_loader, device,
        local_epochs=local_epochs, learning_rate=learning_rate, mu=0.01
    )
    results['FedProx'] = fedprox.train(num_rounds, client_fraction)
    
    # 3. FedNoLowe
    print("\n" + "=" * 70)
    print("Training with FedNoLowe")
    print("=" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_nolowe = FedNoLowe(
        model, client_loaders, test_loader, device,
        local_epochs=local_epochs, learning_rate=learning_rate
    )
    results['FedNoLowe'] = fed_nolowe.train(num_rounds, client_fraction)
    
    # 4. FedLWS
    print("\n" + "=" * 70)
    print("Training with FedLWS")
    print("=" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_lws = FedLWS(
        model, client_loaders, test_loader, device,
        local_epochs=local_epochs, learning_rate=learning_rate
    )
    results['FedLWS'] = fed_lws.train(num_rounds, client_fraction)
    
    # 5. FedAWA
    print("\n" + "=" * 70)
    print("Training with FedAWA")
    print("=" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_awa = FedAWA(
        model, client_loaders, test_loader, device,
        local_epochs=local_epochs, learning_rate=learning_rate
    )
    results['FedAWA'] = fed_awa.train(num_rounds, client_fraction)
    
    # Print final results
    print("\n" + "=" * 70)
    print(f"Final Results for {dataset_name.upper()}")
    print("=" * 70)
    print(f"{'Method':<20} {'Final Acc':<12} {'Max Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for method_name, history in results.items():
        final_acc = history['accuracy'][-1]
        final_loss = history['loss'][-1]
        max_acc = max(history['accuracy'])
        print(f"{method_name:<20} {final_acc:>10.2f}%  {max_acc:>10.2f}%  {final_loss:>10.4f}")
    
    # Plot comparison
    print(f"\nGenerating comparison plots for {dataset_name.upper()}...")
    plot_comparison(results, dataset_name)
    
    return results


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    num_clients = 10
    num_rounds = 20
    local_epochs = 5
    learning_rate = 0.01
    batch_size = 32
    client_fraction = 1.0
    iid = True
    
    # Run experiments on both datasets
    all_results = {}
    
    # MNIST
    print("\n" + "=" * 70)
    print("STARTING MNIST EXPERIMENTS")
    print("=" * 70)
    all_results['mnist'] = run_experiment(
        dataset_name='mnist',
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        client_fraction=client_fraction,
        iid=iid
    )
    
    # CIFAR10
    print("\n" + "=" * 70)
    print("STARTING CIFAR10 EXPERIMENTS")
    print("=" * 70)
    all_results['cifar10'] = run_experiment(
        dataset_name='cifar10',
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        client_fraction=client_fraction,
        iid=iid
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} - Best accuracy per method:")
        for method_name, history in results.items():
            max_acc = max(history['accuracy'])
            print(f"  {method_name:<20}: {max_acc:.2f}%")
    
    print("\nAll experiments completed successfully!")


if __name__ == '__main__':
    main()
