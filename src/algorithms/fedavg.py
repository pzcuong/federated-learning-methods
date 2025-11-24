"""
Federated Averaging (FedAvg) algorithm.
"""
import torch
import copy
from src.utils.training_utils import train_local_model, aggregate_models, evaluate_model


class FedAvg:
    """Federated Averaging algorithm."""
    
    def __init__(self, model, client_loaders, test_loader, device, 
                 local_epochs=5, learning_rate=0.01):
        """
        Initialize FedAvg algorithm.
        
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
        
        # Train local models
        client_models = []
        client_weights = []
        
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
            
            client_models.append(local_model)
            client_weights.append(len(self.client_loaders[client_id].dataset))
        
        # Aggregate models
        self.global_model = aggregate_models(self.global_model, client_models, client_weights)
        
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
