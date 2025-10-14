# Quick Reference Guide

## Installation

```bash
git clone https://github.com/pzcuong/federated-learning-methods.git
cd federated-learning-methods
pip install -r requirements.txt
```

## Quick Start

### Run Full Comparison (requires internet for MNIST download)
```bash
python compare_methods.py
```

### Run Demo with Synthetic Data (no internet required)
```bash
python demo_synthetic.py
```

### Run Single Algorithm Example
```bash
python example_usage.py
```

## File Overview

| File | Purpose |
|------|---------|
| `compare_methods.py` | Compare multiple FL algorithms on MNIST |
| `example_usage.py` | Example of using a single algorithm |
| `demo_synthetic.py` | Demo with synthetic data (for testing) |
| `src/algorithms/fedavg.py` | FedAvg implementation |
| `src/algorithms/fedprox.py` | FedProx implementation |
| `src/models/mnist_net.py` | CNN model for MNIST |
| `src/data/data_loader.py` | Data loading and federated splits |
| `src/utils/training_utils.py` | Training and evaluation utilities |

## Key Parameters

### Federated Learning Configuration
- `num_clients`: Number of federated clients (default: 10)
- `num_rounds`: Communication rounds (default: 20)
- `local_epochs`: Local training epochs per round (default: 5)
- `learning_rate`: Learning rate (default: 0.01)
- `batch_size`: Batch size (default: 32)
- `client_fraction`: Fraction of clients selected per round (default: 1.0)
- `iid`: IID vs non-IID data distribution (default: True)

### FedProx Specific
- `mu`: Proximal term coefficient (default: 0.01)
  - Higher values keep local models closer to global model
  - Useful for non-IID data

## Common Use Cases

### Compare IID vs Non-IID
```python
# In compare_methods.py, change:
iid = False  # For non-IID comparison
```

### Adjust Client Participation
```python
# Select 50% of clients per round:
client_fraction = 0.5
```

### Test Different Model Architectures
```python
# Create your model in src/models/
from src.models.my_model import MyModel
model = MyModel()
```

## Expected Results (Real MNIST)

With default settings after 20 rounds:
- FedAvg: ~98% accuracy
- FedProx (μ=0.01): ~98% accuracy
- FedProx (μ=0.1): ~97-98% accuracy

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/federated-learning-methods
python compare_methods.py
```

### MNIST Download Issues
The first run downloads MNIST (~10MB). Ensure you have:
- Internet connection
- Write permissions in the project directory

### Memory Issues
Reduce the number of clients or batch size:
```python
num_clients = 5  # Instead of 10
batch_size = 16  # Instead of 32
```

## Citation

If you use this code in your research:

```bibtex
@misc{federated-learning-methods,
  title={Federated Learning Methods Comparison},
  author={Your Name},
  year={2024},
  url={https://github.com/pzcuong/federated-learning-methods}
}
```
