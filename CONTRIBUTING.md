# Contributing to Federated Learning Methods

Thank you for your interest in contributing to this project! This guide will help you understand how to extend and improve the codebase.

## Project Structure

```
federated-learning-methods/
├── src/
│   ├── algorithms/          # Federated learning algorithms
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Neural network models
│   └── utils/              # Utility functions
├── compare_methods.py      # Main comparison script
├── example_usage.py        # Example usage
└── requirements.txt        # Dependencies
```

## Adding a New Federated Learning Algorithm

1. Create a new file in `src/algorithms/` (e.g., `my_algorithm.py`)

2. Implement a class with the following structure:

```python
import torch
import copy
from src.utils.training_utils import train_local_model, aggregate_models, evaluate_model

class MyAlgorithm:
    def __init__(self, model, client_loaders, test_loader, device, 
                 local_epochs=5, learning_rate=0.01, **kwargs):
        self.global_model = model.to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.num_clients = len(client_loaders)
        # Add your algorithm-specific parameters
        
    def train_round(self, client_fraction=1.0):
        """Execute one round of federated learning."""
        # Your algorithm logic here
        # 1. Select clients
        # 2. Train local models
        # 3. Aggregate models
        # 4. Evaluate
        accuracy, loss = evaluate_model(self.global_model, self.test_loader, self.device)
        return accuracy, loss
    
    def train(self, num_rounds, client_fraction=1.0):
        """Train the federated model."""
        history = {'accuracy': [], 'loss': []}
        for round_num in range(num_rounds):
            accuracy, loss = self.train_round(client_fraction)
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            print(f"Round {round_num + 1}/{num_rounds} - "
                  f"Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        return history
```

3. Add your algorithm to `compare_methods.py`:

```python
from src.algorithms.my_algorithm import MyAlgorithm

# In the main() function:
model_my_algo = MNISTNet()
my_algo = MyAlgorithm(
    model_my_algo,
    client_loaders,
    test_loader,
    device,
    local_epochs=local_epochs,
    learning_rate=learning_rate
)
results['My Algorithm'] = my_algo.train(num_rounds, client_fraction)
```

## Adding a New Model Architecture

1. Create a new file in `src/models/` (e.g., `my_model.py`)

2. Implement your model as a PyTorch `nn.Module`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        
    def forward(self, x):
        # Define forward pass
        return x
```

3. Use your model in the comparison script:

```python
from src.models.my_model import MyModel

model = MyModel()
```

## Using Different Datasets

1. Add a new data loading function in `src/data/data_loader.py`:

```python
def load_my_dataset(data_dir='./data'):
    """Load custom dataset."""
    # Your data loading logic
    return train_dataset, test_dataset
```

2. Update the comparison script to use your dataset:

```python
train_dataset, test_dataset = load_my_dataset()
```

## Testing Your Changes

1. Test imports:
```bash
python -c "from src.algorithms.my_algorithm import MyAlgorithm; print('Import successful')"
```

2. Test with minimal configuration:
```bash
python example_usage.py
```

3. Run full comparison:
```bash
python compare_methods.py
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Reporting Issues

If you encounter bugs or have feature requests:
1. Check if the issue already exists
2. Provide a clear description
3. Include code to reproduce the issue
4. Specify your environment (Python version, OS, etc.)

## Research Citations

If you use this project in your research, please cite the relevant papers:
- FedAvg: McMahan et al., 2017
- FedProx: Li et al., 2020
