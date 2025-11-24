"""
FedNoLowe (Federated Normalized Loss-based Weighted Aggregation) algorithm.

This algorithm uses client training losses to determine aggregation weights.
Clients with lower training loss (indicating better learning) are given higher weight.
The weighting scheme uses a two-stage normalization process.

Reference: Based on the implementation provided in referrence/fed_nolowe.py
"""
import torch
import copy
import numpy as np
from src.utils.training_utils import train_local_model, evaluate_model


class FedNoLowe:
    """
    Federated Normalized Loss-based Weighted Aggregation (FedNoLowe) algorithm.
    """
    
    def __init__(self, model, client_loaders, test_loader, device, 
                 local_epochs=5, learning_rate=0.01):
        """
        Initialize FedNoLowe algorithm.
        
        Args:
            model: The global model
            client_loaders: List of data loaders for clients
            test_loader: DataLoader for test data
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
        """
        self.global_model = model.to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.num_clients = len(client_loaders)
        
    def calculate_weights(self, train_losses):
        """
        Calculate aggregation weights based on training losses.
        
        Logic from reference implementation:
        1. Normalize losses to sum to 1
        2. Invert probabilities (1 - p)
        3. Normalize again to sum to 1
        
        Args:
            train_losses: List of training losses from clients
            
        Returns:
            List of weights summing to 1
        """
        losses = np.array(train_losses, dtype=np.float64)
        
        # Handle edge case where sum is 0 (all perfect training or empty)
        if losses.sum() == 0:
            return np.ones(len(losses)) / len(losses)
            
        # First normalization
        normalized_losses = losses / losses.sum()
        
        # Inversion (higher loss -> lower weight)
        inverted_weights = 1.0 - normalized_losses
        
        # Second normalization
        if inverted_weights.sum() == 0:
             # This happens if there's only 1 client (1-1=0) or weird distribution
             return np.ones(len(losses)) / len(losses)
             
        final_weights = inverted_weights / inverted_weights.sum()
        
        return final_weights
    
    def aggregate_models(self, global_model, client_models, weights):
        """
        Aggregate client models using calculated weights.
        
        Args:
            global_model: The global model to update
            client_models: List of client models
            weights: List of weights for each client
            
        Returns:
            Updated global model
        """
        weighted_sum = {
            key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) 
            for key in global_model.state_dict().keys()
        }
        
        for key in global_model.state_dict().keys():
            for i, client_model in enumerate(client_models):
                weighted_sum[key] += client_model.state_dict()[key].float() * weights[i]
                
        global_model.load_state_dict(weighted_sum)
        return global_model
        
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning.
        
        Args:
            client_fraction: Fraction of clients to select for training
            
        Returns:
            accuracy: Test accuracy after this round
            loss: Test loss after this round
        """
        # Select clients
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
        
        client_models = []
        client_losses = []
        
        for client_id in selected_clients:
            # Create local model (copy of global model)
            local_model = copy.deepcopy(self.global_model)
            
            # Train on client data and get training loss
            local_model, train_loss = train_local_model(
                local_model,
                self.client_loaders[client_id],
                self.local_epochs,
                self.learning_rate,
                self.device
            )
            
            client_models.append(local_model)
            client_losses.append(train_loss)
        
        # Calculate weights based on training losses
        weights = self.calculate_weights(client_losses)
        
        # Aggregate models
        self.global_model = self.aggregate_models(self.global_model, client_models, weights)
        
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
