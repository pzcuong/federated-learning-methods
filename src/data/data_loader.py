"""
Data loading and preprocessing for federated learning.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def load_mnist(data_dir='./data'):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def load_cifar10(data_dir='./data'):
    """Load CIFAR10 dataset."""
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Simple normalization for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset


def load_dataset(dataset_name='mnist', data_dir='./data'):
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of dataset ('mnist' or 'cifar10')
        data_dir: Directory to store/load data
        
    Returns:
        train_dataset, test_dataset
    """
    if dataset_name.lower() == 'mnist':
        return load_mnist(data_dir)
    elif dataset_name.lower() == 'cifar10':
        return load_cifar10(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'mnist' or 'cifar10'.")


def create_federated_data(dataset, num_clients, iid=True):
    """
    Split dataset into federated clients.
    
    Args:
        dataset: The dataset to split
        num_clients: Number of federated clients
        iid: If True, data is distributed IID; otherwise non-IID
        
    Returns:
        List of client datasets
    """
    total_samples = len(dataset)
    
    if iid:
        # IID split: randomly shuffle and split evenly
        indices = np.random.permutation(total_samples)
        split_size = total_samples // num_clients
        client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(num_clients)]
    else:
        # Non-IID split: sort by label and distribute
        labels = np.array([dataset[i][1] for i in range(total_samples)])
        sorted_indices = np.argsort(labels)
        
        # Divide into shards
        num_shards = num_clients * 2
        shard_size = total_samples // num_shards
        shards = [sorted_indices[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
        
        # Assign 2 shards to each client
        client_indices = []
        for i in range(num_clients):
            client_data = np.concatenate([shards[i*2], shards[i*2+1]])
            client_indices.append(client_data)
    
    # Create client datasets
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets


def get_data_loaders(client_datasets, test_dataset, batch_size=32):
    """
    Create data loaders for clients and test set.
    
    Args:
        client_datasets: List of client datasets
        test_dataset: Test dataset
        batch_size: Batch size for training
        
    Returns:
        client_loaders: List of data loaders for clients
        test_loader: Data loader for test set
    """
    client_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in client_datasets
    ]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader
