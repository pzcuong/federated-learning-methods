"""
Utility functions for federated learning.
"""
import torch
import torch.nn.functional as F
import copy


def train_local_model(model, data_loader, epochs, learning_rate, device, mu=0.0, global_model=None):
    """
    Train a local model on client data.
    
    Args:
        model: The model to train
        data_loader: DataLoader for the client
        epochs: Number of local epochs
        learning_rate: Learning rate
        device: Device to train on
        mu: Proximal term coefficient (for FedProx)
        global_model: Global model (for FedProx)
        
    Returns:
        Trained model
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Add proximal term for FedProx
            if mu > 0 and global_model is not None:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return model, avg_loss


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
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
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    return accuracy, test_loss


def aggregate_models(global_model, client_models, client_weights=None):
    """
    Aggregate client models using weighted averaging (FedAvg).
    
    Args:
        global_model: The global model to update
        client_models: List of client models
        client_weights: Optional weights for each client (defaults to equal weights)
        
    Returns:
        Updated global model
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    # Aggregate parameters
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
        for client_model, weight in zip(client_models, client_weights):
            global_dict[key] += weight * client_model.state_dict()[key]
    
    global_model.load_state_dict(global_dict)
    
    return global_model


def get_model_params(model):
    """Get model parameters as a list."""
    return [param.data.clone() for param in model.parameters()]


def set_model_params(model, params):
    """Set model parameters from a list."""
    for param, new_param in zip(model.parameters(), params):
        param.data = new_param.clone()
