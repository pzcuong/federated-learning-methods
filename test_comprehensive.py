"""
Comprehensive test comparing all algorithms under different data distributions.
Tests both IID (balanced) and Non-IID (imbalanced) scenarios.
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


def plot_iid_vs_noniid(iid_results, noniid_results, dataset_name, save_path=None):
    """Plot comparison between IID and Non-IID results."""
    if save_path is None:
        save_path = f'{dataset_name}_iid_vs_noniid.png'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # IID Accuracy
    ax = axes[0, 0]
    for method_name, history in iid_results.items():
        ax.plot(history['accuracy'], label=method_name, marker='o', markersize=4)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'IID (Balanced) - Test Accuracy ({dataset_name.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Non-IID Accuracy
    ax = axes[0, 1]
    for method_name, history in noniid_results.items():
        ax.plot(history['accuracy'], label=method_name, marker='o', markersize=4)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'Non-IID (Imbalanced) - Test Accuracy ({dataset_name.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IID Loss
    ax = axes[1, 0]
    for method_name, history in iid_results.items():
        ax.plot(history['loss'], label=method_name, marker='s', markersize=4)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Loss')
    ax.set_title(f'IID (Balanced) - Test Loss ({dataset_name.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Non-IID Loss
    ax = axes[1, 1]
    for method_name, history in noniid_results.items():
        ax.plot(history['loss'], label=method_name, marker='s', markersize=4)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Loss')
    ax.set_title(f'Non-IID (Imbalanced) - Test Loss ({dataset_name.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {save_path}")
    plt.close()


def run_experiments(dataset_name, iid, num_clients=5, num_rounds=50, 
                    local_epochs=5, learning_rate=0.01, batch_size=32):
    """Run all algorithms on a dataset with specified distribution."""
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    distribution = "IID (Balanced)" if iid else "Non-IID (Imbalanced)"
    
    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - {distribution}")
    print("=" * 70)
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | Device: {device}")
    print("=" * 70)
    
    # Load and prepare data
    print(f"Loading {dataset_name.upper()} dataset...")
    train_dataset, test_dataset = load_dataset(dataset_name)
    client_datasets = create_federated_data(train_dataset, num_clients, iid=iid)
    client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, batch_size)
    
    print(f"✓ Data loaded with {distribution} distribution")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Samples per client: ~{len(train_dataset) // num_clients}")
    
    results = {}
    
    # 1. FedAvg
    print("\n" + "-" * 70)
    print("Training FedAvg...")
    print("-" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fedavg = FedAvg(model, client_loaders, test_loader, device,
                   local_epochs=local_epochs, learning_rate=learning_rate)
    results['FedAvg'] = fedavg.train(num_rounds, client_fraction=1.0)
    
    # 2. FedProx
    print("\n" + "-" * 70)
    print("Training FedProx...")
    print("-" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fedprox = FedProx(model, client_loaders, test_loader, device,
                     local_epochs=local_epochs, learning_rate=learning_rate, mu=0.01)
    results['FedProx'] = fedprox.train(num_rounds, client_fraction=1.0)
    
    # 3. FedNoLowe
    print("\n" + "-" * 70)
    print("Training FedNoLowe...")
    print("-" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_nolowe = FedNoLowe(model, client_loaders, test_loader, device,
                          local_epochs=local_epochs, learning_rate=learning_rate)
    results['FedNoLowe'] = fed_nolowe.train(num_rounds, client_fraction=1.0)
    
    # 4. FedLWS
    print("\n" + "-" * 70)
    print("Training FedLWS...")
    print("-" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_lws = FedLWS(model, client_loaders, test_loader, device,
                    local_epochs=local_epochs, learning_rate=learning_rate)
    results['FedLWS'] = fed_lws.train(num_rounds, client_fraction=1.0)
    
    # 5. FedAWA
    print("\n" + "-" * 70)
    print("Training FedAWA...")
    print("-" * 70)
    model = UnifiedCNN(dataset=dataset_name)
    fed_awa = FedAWA(model, client_loaders, test_loader, device,
                    local_epochs=local_epochs, learning_rate=learning_rate)
    results['FedAWA'] = fed_awa.train(num_rounds, client_fraction=1.0)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Results Summary - {dataset_name.upper()} ({distribution})")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Final Acc':<12} {'Max Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for method_name, history in results.items():
        final_acc = history['accuracy'][-1]
        max_acc = max(history['accuracy'])
        final_loss = history['loss'][-1]
        print(f"{method_name:<15} {final_acc:>10.2f}%  {max_acc:>10.2f}%  {final_loss:>10.4f}")
    
    return results


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    num_clients = 5
    num_rounds = 20
    local_epochs = 5
    learning_rate = 0.01
    batch_size = 64
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE FEDERATED LEARNING COMPARISON")
    print("=" * 70)
    print("Testing both IID (balanced) and Non-IID (imbalanced) distributions")
    print(f"Configuration: {num_clients} clients, {num_rounds} rounds")
    print("=" * 70)
    
    # Store all results
    all_results = {}
    
    # ====================================================================
    # MNIST Experiments
    # ====================================================================
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 23 + "MNIST EXPERIMENTS" + " " * 28 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # MNIST - IID
    mnist_iid = run_experiments('mnist', iid=True, num_clients=num_clients,
                               num_rounds=num_rounds, local_epochs=local_epochs,
                               learning_rate=learning_rate, batch_size=batch_size)
    
    # MNIST - Non-IID
    mnist_noniid = run_experiments('mnist', iid=False, num_clients=num_clients,
                                  num_rounds=num_rounds, local_epochs=local_epochs,
                                  learning_rate=learning_rate, batch_size=batch_size)
    
    all_results['mnist'] = {'iid': mnist_iid, 'noniid': mnist_noniid}
    
    # Generate MNIST comparison plot
    plot_iid_vs_noniid(mnist_iid, mnist_noniid, 'mnist')
    
    # ====================================================================
    # CIFAR10 Experiments
    # ====================================================================
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 22 + "CIFAR10 EXPERIMENTS" + " " * 27 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # CIFAR10 - IID
    cifar10_iid = run_experiments('cifar10', iid=True, num_clients=num_clients,
                                 num_rounds=num_rounds, local_epochs=local_epochs,
                                 learning_rate=learning_rate, batch_size=batch_size)
    
    # CIFAR10 - Non-IID
    cifar10_noniid = run_experiments('cifar10', iid=False, num_clients=num_clients,
                                    num_rounds=num_rounds, local_epochs=local_epochs,
                                    learning_rate=learning_rate, batch_size=batch_size)
    
    all_results['cifar10'] = {'iid': cifar10_iid, 'noniid': cifar10_noniid}
    
    # Generate CIFAR10 comparison plot
    plot_iid_vs_noniid(cifar10_iid, cifar10_noniid, 'cifar10')
    
    # ====================================================================
    # Final Comprehensive Summary
    # ====================================================================
    print("\n\n" + "=" * 70)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("=" * 70)
    
    for dataset_name in ['mnist', 'cifar10']:
        print(f"\n{dataset_name.upper()}:")
        print("-" * 70)
        
        # IID Results
        print(f"\n  IID (Balanced) - Best Accuracy:")
        for method_name, history in all_results[dataset_name]['iid'].items():
            max_acc = max(history['accuracy'])
            final_acc = history['accuracy'][-1]
            print(f"    {method_name:<15}: Max={max_acc:>6.2f}%  Final={final_acc:>6.2f}%")
        
        # Non-IID Results
        print(f"\n  Non-IID (Imbalanced) - Best Accuracy:")
        for method_name, history in all_results[dataset_name]['noniid'].items():
            max_acc = max(history['accuracy'])
            final_acc = history['accuracy'][-1]
            print(f"    {method_name:<15}: Max={max_acc:>6.2f}%  Final={final_acc:>6.2f}%")
        
        # Compare IID vs Non-IID degradation
        print(f"\n  Performance Impact (IID → Non-IID):")
        for method_name in all_results[dataset_name]['iid'].keys():
            iid_max = max(all_results[dataset_name]['iid'][method_name]['accuracy'])
            noniid_max = max(all_results[dataset_name]['noniid'][method_name]['accuracy'])
            degradation = iid_max - noniid_max
            print(f"    {method_name:<15}: {degradation:>+6.2f}% degradation")
    
    print("\n" + "=" * 70)
    print("✓ All experiments completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - mnist_iid_vs_noniid.png")
    print("  - cifar10_iid_vs_noniid.png")
    print("\nThese plots compare all algorithms under both data distributions.")


if __name__ == '__main__':
    main()
