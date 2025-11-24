"""
FedAWA (Federated learning with Adaptive Weight Aggregation) algorithm.

This is an improved implementation based on the official CVPR 2025 paper.
Reference: https://arxiv.org/abs/2503.15842

The algorithm adaptively optimizes aggregation weights using client vectors
to improve performance with heterogeneous (non-IID) data.
"""
import torch
import copy
import numpy as np
from src.utils.training_utils import train_local_model, evaluate_model


class FedAWA:
    """
    Federated learning with Adaptive Weight Aggregation algorithm.
    
    Based on the official implementation from CVPR 2025 paper.
    Key improvements:
    - Uses softmax for probability distribution over weights
    - Optimizes based on L2 distance between client gradients and weighted sum
    - Adds cosine distance regularization term
    - Supports gamma scaling parameter
    """
    
    def __init__(self, model, client_loaders, test_loader, device,
                 local_epochs=5, learning_rate=0.01, 
                 server_epochs=1, server_optimizer='adam',
                 gamma=1.0, reg_distance='cos'):
        """
        Initialize FedAWA algorithm.
        
        Args:
            model: The global model
            client_loaders: List of data loaders for clients
            test_loader: DataLoader for test data
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            server_epochs: Number of server-side weight optimization epochs
            server_optimizer: Optimizer for weight adaptation ('adam' or 'sgd')
            gamma: Scaling factor for aggregation (default: 1.0)
            reg_distance: Distance metric for regularization ('cos' or 'euc')
        """
        self.global_model = model.to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.num_clients = len(client_loaders)
        self.server_epochs = server_epochs
        self.server_optimizer = server_optimizer
        self.gamma = gamma
        self.reg_distance = reg_distance
        
        # Initialize aggregation weights (will be updated adaptively)
        self.agg_weights = torch.ones(self.num_clients, device=device) / self.num_clients
        
    def flatten_model(self, model):
        """Flatten model parameters into a single vector."""
        param_list = []
        for param in model.parameters():
            param_list.append(param.data.flatten())
        return torch.cat(param_list)
    
    def compute_client_update(self, global_params, local_params):
        """
        Compute the update (gradient) from global to local parameters.
        
        Args:
            global_params: Flattened global model parameters
            local_params: Flattened local model parameters
            
        Returns:
            Client update vector
        """
        return local_params - global_params
    
    def cost_matrix(self, x, y):
        """
        Compute pairwise distance matrix between vectors.
        
        Args:
            x: Tensor of shape (1, D) - global parameters
            y: Tensor of shape (N, D) - local parameters
            
        Returns:
            Cost matrix of shape (1, N)
        """
        x_col = x.unsqueeze(-2)  # (1, 1, D)
        y_lin = y.unsqueeze(-3)  # (1, N, D)
        
        if self.reg_distance == 'cos':
            # Cosine distance: 1 - cosine_similarity
            d_cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
            C = 1 - d_cosine(x_col, y_lin)
        elif self.reg_distance == 'euc':
            # Euclidean distance
            C = torch.mean(torch.abs(x_col - y_lin) ** 2, -1)
        else:
            raise ValueError(f"Unknown distance metric: {self.reg_distance}")
        
        return C
    
    def optimize_aggregation_weights(self, global_params, client_params_list):
        """
        Optimize aggregation weights using the FedAWA algorithm.
        
        This follows the official implementation:
        1. Use softmax to ensure weights form a probability distribution
        2. Minimize L2 distance between individual client gradients and weighted sum
        3. Add regularization based on distance from global model
        
        Args:
            global_params: Flattened global model parameters
            client_params_list: List of flattened client model parameters
            
        Returns:
            Optimized probability distribution over clients
        """
        # Stack client parameters
        local_param_tensor = torch.stack(client_params_list)  # (N, D)
        
        # Initialize weights with requires_grad
        T_weights = self.agg_weights.clone().detach().requires_grad_(True)
        
        # Setup optimizer
        if self.server_optimizer == 'sgd':
            optimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif self.server_optimizer == 'adam':
            optimizer = torch.optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))
        else:
            raise ValueError(f"Unknown optimizer: {self.server_optimizer}")
        
        # Optimize weights for server_epochs iterations
        for epoch in range(self.server_epochs):
            optimizer.zero_grad()
            
            # Apply softmax to get probability distribution
            probability = torch.nn.functional.softmax(T_weights, dim=0)
            
            # Regularization term: weighted distance from global model to local models
            C = self.cost_matrix(global_params.unsqueeze(0), local_param_tensor)
            reg_loss = torch.sum(probability * C)
            
            # Compute client updates (gradients)
            client_grads = local_param_tensor - global_params  # (N, D)
            
            # Weighted sum of client gradients
            weighted_grad = torch.matmul(probability.unsqueeze(0), client_grads)  # (1, D)
            
            # Similarity loss: L2 distance between each client grad and the weighted sum
            l2_distance = torch.norm(client_grads.unsqueeze(0) - weighted_grad.unsqueeze(1), 
                                    p=2, dim=2)  # (1, N)
            sim_loss = torch.sum(probability * l2_distance)
            
            # Total loss
            loss = sim_loss + reg_loss
            
            # Backprop and update
            loss.backward()
            optimizer.step()
        
        # Return final probability distribution
        with torch.no_grad():
            final_prob = torch.nn.functional.softmax(T_weights, dim=0)
            self.agg_weights = T_weights.data  # Store raw weights for next round
        
        return final_prob
    
    def aggregate_with_adaptive_weights(self, global_model, client_models, weights):
        """
        Aggregate client models using adaptive weights with gamma scaling.
        
        Args:
            global_model: The global model to update
            client_models: List of client models
            weights: Adaptive probability weights for each client
            
        Returns:
            Updated global model
        """
        global_dict = global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_model, weight in zip(client_models, weights):
                # Apply gamma scaling as in the reference implementation
                global_dict[key] += weight * client_model.state_dict()[key] * self.gamma
            # Normalize by sum of probabilities (which should be 1)
            global_dict[key] /= weights.sum()
        
        global_model.load_state_dict(global_dict)
        return global_model
        
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning with adaptive weight aggregation.
        
        Args:
            client_fraction: Fraction of clients to select for training
            
        Returns:
            accuracy: Test accuracy after this round
            loss: Test loss after this round
        """
        # Select clients
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
        
        # Flatten global model parameters
        global_params = self.flatten_model(self.global_model)
        
        # Train local models and collect parameters
        client_models = []
        client_params_list = []
        
        for client_id in selected_clients:
            # Create local model (copy of global model)
            local_model = copy.deepcopy(self.global_model)
            
            # Train on client data
            local_model, _ = train_local_model(
                local_model,
                self.client_loaders[client_id],
                self.local_epochs,
                self.learning_rate,
                self.device
            )
            
            # Flatten local parameters
            local_params = self.flatten_model(local_model)
            
            client_models.append(local_model)
            client_params_list.append(local_params)
        
        # Optimize aggregation weights
        adaptive_weights = self.optimize_aggregation_weights(global_params, client_params_list)
        
        # Aggregate models with adaptive weights
        self.global_model = self.aggregate_with_adaptive_weights(
            self.global_model, client_models, adaptive_weights
        )
        
        # Evaluate
        accuracy, loss = evaluate_model(self.global_model, self.test_loader, self.device)
        
        return accuracy, loss
    
    def train(self, num_rounds, client_fraction=1.0):
        """
        Train the federated model.
        
        Args:
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to select per round
            
        Returns:
            history: Dictionary with training history
        """
        history = {
            'accuracy': [],
            'loss': []
        }
        
        for round_num in range(num_rounds):
            accuracy, loss = self.train_round(client_fraction)
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            
            print(f"Round {round_num + 1}/{num_rounds} - "
                  f"Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        
        return history
