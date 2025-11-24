"""
FedLWS (Federated Learning with Adaptive Layer-wise Weight Shrinking) algorithm.

This algorithm dynamically calculates layer-wise shrinking factors based on
local gradient variances to improve generalization without requiring proxy datasets.
"""
import torch
import copy
import numpy as np
from src.utils.training_utils import evaluate_model


class FedLWS:
    """Federated Learning with Adaptive Layer-wise Weight Shrinking algorithm."""
    
    def __init__(self, model, client_loaders, test_loader, device,
                 local_epochs=5, learning_rate=0.01):
        """
        Initialize FedLWS algorithm.
        
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
        self.gradient_variances = None
        
    def train_local_and_collect_gradients(self, model, data_loader):
        """
        Train local model and collect gradient information.
        
        Args:
            model: The model to train
            data_loader: DataLoader for the client
            
        Returns:
            Trained model and gradient variances per layer
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        
        # Store gradients for variance calculation
        all_gradients = {name: [] for name, _ in model.named_parameters()}
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                # Collect gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        all_gradients[name].append(param.grad.clone().detach())
                
                optimizer.step()
        
        # Calculate variance for each layer
        gradient_variances = {}
        for name, grads in all_gradients.items():
            if len(grads) > 0:
                grads_tensor = torch.stack(grads)
                gradient_variances[name] = torch.var(grads_tensor, dim=0).mean().item()
            else:
                gradient_variances[name] = 0.0
        
        return model, gradient_variances
        
    def compute_shrinking_factors(self, all_variances):
        """
        Compute layer-wise shrinking factors based on gradient variances.
        
        Args:
            all_variances: List of variance dictionaries from all clients
            
        Returns:
            Dictionary of shrinking factors per layer
        """
        if not all_variances:
            return {}
        
        # Average variances across clients
        layer_names = all_variances[0].keys()
        avg_variances = {}
        
        for name in layer_names:
            variances = [client_vars[name] for client_vars in all_variances]
            avg_variances[name] = np.mean(variances)
        
        # Compute shrinking factors: higher variance -> smaller shrinking factor
        # Using formula: shrinking_factor = 1 / (1 + variance)
        shrinking_factors = {}
        for name, variance in avg_variances.items():
            shrinking_factors[name] = 1.0 / (1.0 + variance)
        
        return shrinking_factors
    
    def aggregate_with_shrinking(self, global_model, client_models, client_weights, shrinking_factors):
        """
        Aggregate client models with layer-wise weight shrinking.
        
        Args:
            global_model: The global model to update
            client_models: List of client models
            client_weights: Weights for each client (based on dataset size)
            shrinking_factors: Dictionary of shrinking factors per layer
            
        Returns:
            Updated global model
        """
        # Normalize client weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        global_dict = global_model.state_dict()
        
        # First, perform standard weighted aggregation
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for client_model, weight in zip(client_models, normalized_weights):
                global_dict[key] += weight * client_model.state_dict()[key]
        
        # Then apply layer-wise shrinking
        if shrinking_factors:
            for key in global_dict.keys():
                # Find matching shrinking factor (key might have .weight or .bias suffix)
                base_key = key.replace('.weight', '').replace('.bias', '')
                for var_name, factor in shrinking_factors.items():
                    if base_key in var_name or var_name in base_key:
                        global_dict[key] *= factor
                        break
        
        global_model.load_state_dict(global_dict)
        return global_model
        
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning with layer-wise weight shrinking.
        
        Args:
            client_fraction: Fraction of clients to select for training
            
        Returns:
            accuracy: Test accuracy after this round
            loss: Test loss after this round
        """
        # Select clients
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
        
        # Train local models and collect gradient variances
        client_models = []
        client_weights = []
        all_variances = []
        
        for client_id in selected_clients:
            # Create local model (copy of global model)
            local_model = copy.deepcopy(self.global_model)
            
            # Train and collect gradient information
            local_model, grad_variances = self.train_local_and_collect_gradients(
                local_model,
                self.client_loaders[client_id]
            )
            
            client_models.append(local_model)
            client_weights.append(len(self.client_loaders[client_id].dataset))
            all_variances.append(grad_variances)
        
        # Compute layer-wise shrinking factors
        shrinking_factors = self.compute_shrinking_factors(all_variances)
        
        # Aggregate models with layer-wise shrinking
        self.global_model = self.aggregate_with_shrinking(
            self.global_model, client_models, client_weights, shrinking_factors
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
