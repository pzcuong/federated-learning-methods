"""
Unified CNN model that works for both MNIST and CIFAR10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCNN(nn.Module):
    """Convolutional Neural Network that adapts to MNIST or CIFAR10."""
    
    def __init__(self, dataset='mnist'):
        """
        Initialize the CNN model.
        
        Args:
            dataset: Either 'mnist' or 'cifar10'
        """
        super(UnifiedCNN, self).__init__()
        self.dataset = dataset.lower()
        
        if self.dataset == 'mnist':
            # MNIST: grayscale 28x28
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, 10)
        elif self.dataset == 'cifar10':
            # CIFAR10: RGB 32x32
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    def forward(self, x):
        if self.dataset == 'mnist':
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.dataset == 'cifar10':
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 256 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        
        return x
