"""
FedCIL (Federated Causal Invariant Learning) Model Architecture.

This model implements a dual-branch encoder architecture:
- Causal Encoder (E_c): Extracts causal features (Z_c) that are invariant across environments
- Environment Encoder (E_e): Extracts environment-specific features (Z_e)

The predictor uses only Z_c for prediction, enforcing causal invariance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedCILModel(nn.Module):
    """
    Federated Causal Invariant Learning model with dual-branch encoder.
    
    Architecture:
    - Shared feature extractor (initial layers)
    - Causal Encoder (E_c) -> Z_c (causal features)
    - Environment Encoder (E_e) -> Z_e (environment features)
    - Predictor: Z_c -> Y (prediction)
    
    The independence loss ensures Z_c and Z_e are orthogonal/independent.
    """
    
    def __init__(self, dataset='mnist', latent_dim=128):
        """
        Initialize the FedCIL model.
        
        Args:
            dataset: Either 'mnist' or 'cifar10'
            latent_dim: Dimension of latent space (Z_c and Z_e)
        """
        super(FedCILModel, self).__init__()
        self.dataset = dataset.lower()
        self.latent_dim = latent_dim
        
        if self.dataset == 'mnist':
            # MNIST: grayscale 28x28
            in_channels = 1
            # Shared feature extractor
            self.shared_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
            self.shared_conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            feature_dim = 64 * 7 * 7
            
            # Causal Encoder (E_c)
            self.causal_encoder = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            
            # Environment Encoder (E_e)
            self.env_encoder = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            
        elif self.dataset == 'cifar10':
            # CIFAR10: RGB 32x32
            in_channels = 3
            # Shared feature extractor
            self.shared_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.shared_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.shared_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            feature_dim = 256 * 4 * 4
            
            # Causal Encoder (E_c)
            self.causal_encoder = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim)
            )
            
            # Environment Encoder (E_e)
            self.env_encoder = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim)
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Predictor (only uses Z_c)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def extract_features(self, x):
        """Extract shared features from input."""
        if self.dataset == 'mnist':
            x = F.relu(self.shared_conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.shared_conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64 * 7 * 7)
        elif self.dataset == 'cifar10':
            x = F.relu(self.shared_conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.shared_conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.shared_conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 256 * 4 * 4)
        return x
    
    def forward(self, x, return_latent=False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            return_latent: If True, also return Z_c and Z_e
            
        Returns:
            output: Predicted logits
            (optional) z_c: Causal latent features
            (optional) z_e: Environment latent features
        """
        # Extract shared features
        features = self.extract_features(x)
        
        # Encode into causal and environment features
        z_c = self.causal_encoder(features)  # Causal features
        z_e = self.env_encoder(features)     # Environment features
        
        # Predict using only causal features
        output = self.predictor(z_c)
        
        if return_latent:
            return output, z_c, z_e
        return output
    
    def get_causal_params(self):
        """
        Get parameters that should be shared globally (causal encoder + predictor).
        
        Returns:
            Dictionary of parameter names to parameters
        """
        causal_params = {}
        
        # Shared feature extractor (shared across all clients)
        for name, param in self.named_parameters():
            if 'shared_' in name:
                causal_params[name] = param
        
        # Causal encoder
        for name, param in self.causal_encoder.named_parameters():
            causal_params[f'causal_encoder.{name}'] = param
        
        # Predictor
        for name, param in self.predictor.named_parameters():
            causal_params[f'predictor.{name}'] = param
        
        return causal_params
    
    def get_env_params(self):
        """
        Get environment encoder parameters (not shared globally).
        
        Returns:
            Dictionary of parameter names to parameters
        """
        env_params = {}
        for name, param in self.env_encoder.named_parameters():
            env_params[f'env_encoder.{name}'] = param
        return env_params
    
    def get_causal_state_dict(self):
        """
        Get state dict for causal components only.
        
        Returns:
            State dict for shared features, causal encoder, and predictor
        """
        full_state = self.state_dict()
        causal_state = {}
        for key, value in full_state.items():
            if 'env_encoder' not in key:
                causal_state[key] = value
        return causal_state
    
    def load_causal_state_dict(self, causal_state):
        """
        Load state dict for causal components only (preserving env encoder).
        
        Args:
            causal_state: State dict containing causal component weights
        """
        current_state = self.state_dict()
        for key, value in causal_state.items():
            if key in current_state:
                current_state[key] = value
        self.load_state_dict(current_state)


def compute_independence_loss(z_c, z_e):
    """
    Compute independence loss between causal and environment features.
    
    Uses the Frobenius norm of the cross-covariance matrix to encourage
    orthogonality between Z_c and Z_e.
    
    L_indep = ||Z_c^T · Z_e / batch_size||_F^2
    
    This is equivalent to ||cov(Z_c, Z_e)||_F^2, measuring the squared
    Frobenius norm of the cross-covariance matrix.
    
    Args:
        z_c: Causal features [batch_size, latent_dim]
        z_e: Environment features [batch_size, latent_dim]
        
    Returns:
        Independence loss (scalar)
    """
    batch_size = z_c.size(0)
    
    # Normalize features (zero-mean)
    z_c_centered = z_c - z_c.mean(dim=0, keepdim=True)
    z_e_centered = z_e - z_e.mean(dim=0, keepdim=True)
    
    # Compute cross-covariance matrix
    # cov = Z_c^T @ Z_e / batch_size
    cross_cov = torch.mm(z_c_centered.t(), z_e_centered) / batch_size
    
    # Frobenius norm squared (sum of squared elements)
    indep_loss = torch.sum(cross_cov ** 2)
    
    return indep_loss


def hsic_loss(z_c, z_e, sigma=1.0):
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) loss.
    
    HSIC measures statistical independence between two random variables.
    Lower HSIC indicates more independence.
    
    Args:
        z_c: Causal features [batch_size, latent_dim]
        z_e: Environment features [batch_size, latent_dim]
        sigma: Kernel bandwidth parameter
        
    Returns:
        HSIC loss (scalar)
    """
    batch_size = z_c.size(0)
    
    # Compute kernel matrices using RBF kernel
    def rbf_kernel(x, sigma):
        xx = torch.mm(x, x.t())
        x_sq = torch.diag(xx)
        dist = x_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2 * xx
        return torch.exp(-dist / (2 * sigma ** 2))
    
    K_c = rbf_kernel(z_c, sigma)
    K_e = rbf_kernel(z_e, sigma)
    
    # Center kernel matrices implicitly (memory-efficient)
    # HKH = K - 1/n * K @ 1 - 1/n * 1 @ K + 1/n^2 * 1 @ K @ 1
    # This simplifies to: K - row_mean - col_mean + total_mean
    K_c_row_mean = K_c.mean(dim=1, keepdim=True)
    K_c_col_mean = K_c.mean(dim=0, keepdim=True)
    K_c_total_mean = K_c.mean()
    K_c_centered = K_c - K_c_row_mean - K_c_col_mean + K_c_total_mean
    
    K_e_row_mean = K_e.mean(dim=1, keepdim=True)
    K_e_col_mean = K_e.mean(dim=0, keepdim=True)
    K_e_total_mean = K_e.mean()
    K_e_centered = K_e - K_e_row_mean - K_e_col_mean + K_e_total_mean
    
    # HSIC = trace(K_c_centered @ K_e_centered) / (batch_size - 1)^2
    hsic = torch.trace(torch.mm(K_c_centered, K_e_centered)) / ((batch_size - 1) ** 2)
    
    return hsic
