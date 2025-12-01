"""
FedCIL (Federated Causal Invariant Learning) algorithm.

This algorithm addresses Non-IID data distribution by separating causal and 
environment-specific features, then only aggregating causal components.

Key components:
1. Dual-branch encoder (Causal + Environment)
2. Dual loss function: L_task (Cross-Entropy) + λ * L_indep (Independence penalty)
3. Selective aggregation: Only aggregate causal encoder and predictor weights

Reference: Based on the mathematical formulation in the problem statement
"""
import torch
import torch.nn.functional as F
import copy
import numpy as np
from src.models.fedcil_model import FedCILModel, compute_independence_loss
from src.utils.training_utils import evaluate_model


class FedCIL:
    """
    Federated Causal Invariant Learning algorithm.
    
    This algorithm enforces separation between causal (invariant) and 
    environment-specific (varying) features across clients.
    
    Optimization objective at each client k:
        min_{θ_c, θ_e, θ_p} L_CE(Predictor(Z_c), Y) + λ * ||Z_c^T · Z_e||_F^2
    
    Only causal components (θ_c, θ_p) are aggregated globally.
    """
    
    def __init__(self, model, client_loaders, test_loader, device,
                 local_epochs=5, learning_rate=0.01, 
                 lambda_indep=0.1, latent_dim=128):
        """
        Initialize FedCIL algorithm.
        
        Args:
            model: The global model (FedCILModel or compatible)
            client_loaders: List of data loaders for clients
            test_loader: DataLoader for test data
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            lambda_indep: Weight for independence loss (default: 0.1)
            latent_dim: Dimension of latent space (default: 128)
        """
        # Determine dataset type from the model or first batch
        if hasattr(model, 'dataset'):
            dataset = model.dataset
        else:
            # Try to infer from data
            sample_batch = next(iter(client_loaders[0]))[0]
            if sample_batch.shape[1] == 1:
                dataset = 'mnist'
            else:
                dataset = 'cifar10'
        
        # Create FedCIL model
        self.global_model = FedCILModel(dataset=dataset, latent_dim=latent_dim).to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.lambda_indep = lambda_indep
        self.num_clients = len(client_loaders)
        self.latent_dim = latent_dim
        self.dataset = dataset
    
    def train_local_model(self, model, data_loader):
        """
        Train local model with dual loss function.
        
        L_total = L_task + λ * L_indep
        
        Args:
            model: The model to train
            data_loader: DataLoader for the client
            
        Returns:
            Trained model
            Average task loss
            Average independence loss
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        
        total_task_loss = 0.0
        total_indep_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass with latent features
                output, z_c, z_e = model(data, return_latent=True)
                
                # Task loss (Cross-Entropy)
                task_loss = F.cross_entropy(output, target)
                
                # Independence loss (orthogonality penalty)
                indep_loss = compute_independence_loss(z_c, z_e)
                
                # Total loss
                total_loss = task_loss + self.lambda_indep * indep_loss
                
                total_loss.backward()
                optimizer.step()
                
                total_task_loss += task_loss.item()
                total_indep_loss += indep_loss.item()
                num_batches += 1
        
        avg_task_loss = total_task_loss / num_batches if num_batches > 0 else 0.0
        avg_indep_loss = total_indep_loss / num_batches if num_batches > 0 else 0.0
        
        return model, avg_task_loss, avg_indep_loss
    
    def aggregate_causal_models(self, global_model, client_models, client_weights):
        """
        Aggregate only causal components from client models.
        
        This implements selective aggregation where:
        - Shared features are averaged
        - Causal encoder is averaged
        - Predictor is averaged
        - Environment encoder is NOT aggregated (kept local)
        
        Args:
            global_model: The global model to update
            client_models: List of client models
            client_weights: Weights for each client (based on dataset size)
            
        Returns:
            Updated global model
        """
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Get causal state dicts from all clients
        client_causal_states = [model.get_causal_state_dict() for model in client_models]
        
        # Initialize aggregated state with zeros
        global_causal_state = {}
        for key in client_causal_states[0].keys():
            global_causal_state[key] = torch.zeros_like(client_causal_states[0][key])
        
        # Weighted average of causal components
        for key in global_causal_state.keys():
            for i, client_state in enumerate(client_causal_states):
                global_causal_state[key] += normalized_weights[i] * client_state[key].float()
        
        # Load aggregated causal state into global model
        global_model.load_causal_state_dict(global_causal_state)
        
        return global_model
    
    def evaluate_model(self, model, test_loader):
        """
        Evaluate model on test data.
        
        Args:
            model: The model to evaluate
            test_loader: DataLoader for test data
            
        Returns:
            accuracy: Test accuracy
            loss: Average test loss
        """
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)  # Only uses causal features for prediction
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        return accuracy, test_loss
    
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning with FedCIL.
        
        Steps:
        1. Select clients
        2. Train local models with dual loss (disentanglement training)
        3. Extract and aggregate causal components only
        4. Evaluate global model
        
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
        total_indep_loss = 0.0
        
        for client_id in selected_clients:
            # Create local model (copy of global model)
            local_model = copy.deepcopy(self.global_model)
            
            # Train with dual loss
            local_model, task_loss, indep_loss = self.train_local_model(
                local_model,
                self.client_loaders[client_id]
            )
            
            client_models.append(local_model)
            client_weights.append(len(self.client_loaders[client_id].dataset))
            total_indep_loss += indep_loss
        
        # Aggregate causal components only
        self.global_model = self.aggregate_causal_models(
            self.global_model, client_models, client_weights
        )
        
        # Evaluate
        accuracy, loss = self.evaluate_model(self.global_model, self.test_loader)
        
        return accuracy, loss
    
    def train(self, num_rounds, client_fraction=1.0):
        """
        Train the federated model using FedCIL.
        
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
