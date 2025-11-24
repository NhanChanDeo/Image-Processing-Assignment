# Training Results & Model Saving Guide

## Overview
The training results saver automatically saves:
- ✅ Best model checkpoint
- ✅ Training/validation history (JSON + CSV)
- ✅ Training plots (curves, epoch comparison)
- ✅ Comprehensive metrics summary
- ✅ Training report

## Directory Structure

After training, `checkpoints/` will contain:

```
checkpoints/
├── models/
│   ├── best_model.pth              # Best model checkpoint
│   ├── checkpoint_epoch_001.pth
│   ├── checkpoint_epoch_002.pth
│   └── ...
├── logs/
│   ├── train_YYYYMMDD_HHMMSS.log  # Detailed training logs
│   ├── training_history.json       # Full training history
│   ├── training_history.csv        # Training history (tabular)
│   ├── metrics_summary.json        # Final metrics summary
│   └── training_report.json        # Comprehensive report
└── plots/
    ├── training_curves.png         # 4-panel training visualization
    └── epoch_comparison.png        # Best vs Final epoch comparison
```

## Usage

### 1. Train Model with Auto-Saving

```bash
python main.py \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 10 \
  --checkpoint_dir ./checkpoints
```

### 2. Load Saved Model

```python
import torch
from model.model import AgeGenderModel
from trainer.save_utils import load_model

# Load best model
model, checkpoint = load_model(
    AgeGenderModel, 
    'checkpoints/models/best_model.pth',
    device='cuda'
)

print(f"Model loaded from epoch: {checkpoint['epoch']}")
print(f"Validation metrics: {checkpoint['metrics']}")
```

### 3. Access Training History

```python
import json
import pandas as pd

# Load training history
with open('checkpoints/logs/training_history.json', 'r') as f:
    history = json.load(f)

# Or as DataFrame
df = pd.read_csv('checkpoints/logs/training_history.csv')
print(df.tail())
```

### 4. View Training Report

```python
import json

with open('checkpoints/logs/training_report.json', 'r') as f:
    report = json.load(f)

print(f"Best epoch: {report['training_summary']['best_epoch']}")
print(f"Age MAE: {report['best_epoch_metrics']['age_mae']:.2f} years")
print(f"Gender Acc: {report['best_epoch_metrics']['gender_accuracy_percent']:.2f}%")
```

## Saved Files Explanation

### `best_model.pth`
PyTorch checkpoint containing:
```python
{
    'epoch': 35,                           # Epoch when saved
    'model_state_dict': {...},             # Model weights
    'optimizer_state_dict': {...},         # Optimizer state
    'metrics': {                           # Validation metrics
        'loss': 25.1234,
        'age_mae': 4.56,
        'gender_accuracy': 0.9678
    },
    'timestamp': '2025-11-24T10:30:45'
}
```

### `training_history.json`
Complete training history with all metrics:
```json
{
    "train_loss": [45.23, 40.12, ...],
    "train_age_loss": [32.12, 28.34, ...],
    "train_gender_loss": [13.11, 11.78, ...],
    "train_age_mae": [12.34, 10.56, ...],
    "train_gender_acc": [0.7845, 0.8234, ...],
    "val_loss": [...],
    "val_age_loss": [...],
    ...
}
```

### `metrics_summary.json`
Final performance summary:
```json
{
    "training_summary": {
        "total_epochs": 35,
        "best_epoch": 25,
        "early_stopped": true
    },
    "best_epoch_metrics": {
        "age_mae": 4.56,
        "gender_accuracy": 0.9678
    },
    "final_epoch_metrics": {
        "age_mae": 4.58,
        "gender_accuracy": 0.9675
    }
}
```

### `training_curves.png`
4-panel visualization:
- Top-left: Total Loss (train vs val)
- Top-right: Component Losses (age & gender)
- Bottom-left: Age MAE
- Bottom-right: Gender Accuracy

### `epoch_comparison.png`
Bar charts comparing Best Epoch vs Final Epoch

## Advanced Usage

### Save Custom Checkpoint

```python
from trainer.save_utils import ResultsSaver

saver = ResultsSaver('./my_checkpoints')

# Save model
model_path = saver.save_model(
    model=model,
    optimizer=optimizer,
    epoch=50,
    metrics={'loss': 25.12, 'acc': 0.96},
    is_best=True
)
```

### Generate Custom Report

```python
config = {
    'max_epochs': 50,
    'patience': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'Adam'
}

report_path, report = saver.create_training_report(
    history=training_history,
    best_epoch=25,
    total_epochs=35,
    config=config
)
```

## Example Workflow

```python
# 1. Train and save
python main.py --epochs 50 --checkpoint_dir ./experiments/exp1

# 2. View training plots
# Open: ./experiments/exp1/plots/training_curves.png

# 3. Load best model
from trainer.save_utils import load_model
model, ckpt = load_model(AgeGenderModel, './experiments/exp1/models/best_model.pth')

# 4. Evaluate on test set
predictions = model(test_images)

# 5. View results
import json
with open('./experiments/exp1/logs/training_report.json') as f:
    report = json.load(f)
print(json.dumps(report, indent=2))
```

## Performance Metrics Meaning

| Metric | Range | Better | Meaning |
|--------|-------|--------|---------|
| **Age MAE** | 0 to ∞ | Lower | Average age prediction error in years |
| **Gender Acc** | 0 to 1 | Higher | Percentage of correct gender predictions |
| **Loss** | 0 to ∞ | Lower | Combined multi-task loss |

## Tips

1. **Early Stopping**: Training stops automatically if val_loss doesn't improve for `patience` epochs
2. **Best Model**: Best model is saved in `models/best_model.pth`
3. **Comparison**: Use `epoch_comparison.png` to verify training didn't overfit
4. **History**: Access full history via JSON for custom analysis
5. **Reproducibility**: Timestamp and config saved for every run

