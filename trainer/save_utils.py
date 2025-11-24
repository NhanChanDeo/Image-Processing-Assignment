"""Utilities for saving training results and models."""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any


class ResultsSaver:
    """Save training results, models, and metrics."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize results saver."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.model_dir = self.checkpoint_dir / "models"
        self.log_dir = self.checkpoint_dir / "logs"
        self.plot_dir = self.checkpoint_dir / "plots"
        
        for directory in [self.model_dir, self.log_dir, self.plot_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: torch.nn.Module, optimizer, epoch: int, 
                   metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save epoch checkpoint
        model_path = self.model_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, model_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            return best_path
        
        return model_path
    
    def save_training_history(self, history: Dict[str, list]):
        """
        Save training history as JSON and CSV.
        
        Args:
            history: Training history dictionary
        """
        # Save as JSON
        json_path = self.log_dir / "training_history.json"
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Save as CSV
        csv_path = self.log_dir / "training_history.csv"
        try:
            import pandas as pd
            df = pd.DataFrame(history)
            df.to_csv(csv_path, index=False)
        except ImportError:
            pass  # pandas not installed
        
        return json_path
    
    def save_metrics_summary(self, summary: Dict[str, Any]):
        """
        Save final metrics summary.
        
        Args:
            summary: Metrics summary dictionary
        """
        summary_path = self.log_dir / "metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        return summary_path
    
    def plot_training_curves(self, history: Dict[str, list]):
        """
        Plot and save training curves.
        
        Args:
            history: Training history dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Loss', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Age Loss
        ax = axes[0, 1]
        ax.plot(epochs, history['train_age_loss'], 'o-', label='Train Age Loss', linewidth=2)
        ax.plot(epochs, history['val_age_loss'], 's-', label='Val Age Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Age Loss (L1)', fontsize=11)
        ax.set_title('Age Loss Component', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gender Loss
        ax = axes[1, 0]
        ax.plot(epochs, history['train_gender_loss'], 'o-', label='Train Gender Loss', linewidth=2)
        ax.plot(epochs, history['val_gender_loss'], 's-', label='Val Gender Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Gender Loss (BCE)', fontsize=11)
        ax.set_title('Gender Loss Component', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Metrics
        ax = axes[1, 1]
        ax.plot(epochs, history['train_age_mae'], 'o-', label='Train Age MAE', linewidth=2)
        ax.plot(epochs, history['val_age_mae'], 's-', label='Val Age MAE', linewidth=2)
        
        ax2 = ax.twinx()
        ax2.plot(epochs, [acc*100 for acc in history['train_gender_acc']], '^-', 
                label='Train Gender Acc %', linewidth=2, color='green')
        ax2.plot(epochs, [acc*100 for acc in history['val_gender_acc']], 'v-', 
                label='Val Gender Acc %', linewidth=2, color='orange')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Age MAE (years)', fontsize=11, color='blue')
        ax2.set_ylabel('Gender Accuracy (%)', fontsize=11, color='green')
        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.plot_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_epoch_comparison(self, history: Dict[str, list], best_epoch: int):
        """
        Plot comparison between best epoch and final epoch.
        
        Args:
            history: Training history
            best_epoch: Best epoch number
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Best Epoch ({best_epoch}) vs Final Epoch ({len(history["train_loss"])})', 
                    fontsize=14, fontweight='bold')
        
        best_idx = best_epoch - 1
        final_idx = len(history['train_loss']) - 1
        
        metrics_names = ['Age MAE', 'Gender Acc %']
        best_vals = [
            history['val_age_mae'][best_idx],
            history['val_gender_acc'][best_idx] * 100
        ]
        final_vals = [
            history['val_age_mae'][final_idx],
            history['val_gender_acc'][final_idx] * 100
        ]
        
        # Plot 1: Metrics comparison
        ax = axes[0]
        x = np.arange(len(metrics_names))
        width = 0.35
        bars1 = ax.bar(x - width/2, best_vals, width, label='Best Epoch', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_vals, width, label='Final Epoch', alpha=0.8)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Validation Metrics Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Loss comparison
        ax = axes[1]
        loss_names = ['Total', 'Age', 'Gender']
        best_losses = [
            history['val_loss'][best_idx],
            history['val_age_loss'][best_idx],
            history['val_gender_loss'][best_idx]
        ]
        final_losses = [
            history['val_loss'][final_idx],
            history['val_age_loss'][final_idx],
            history['val_gender_loss'][final_idx]
        ]
        
        x = np.arange(len(loss_names))
        bars1 = ax.bar(x - width/2, best_losses, width, label='Best Epoch', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_losses, width, label='Final Epoch', alpha=0.8)
        
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.plot_dir / "epoch_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_training_report(self, history: Dict[str, list], best_epoch: int, 
                              total_epochs: int, config: Dict[str, Any]):
        """
        Create comprehensive training report.
        
        Args:
            history: Training history
            best_epoch: Best epoch number
            total_epochs: Total epochs trained
            config: Training configuration
        """
        best_idx = best_epoch - 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_config': config,
            'training_summary': {
                'total_epochs': total_epochs,
                'best_epoch': best_epoch,
                'early_stopped': total_epochs < config.get('max_epochs', total_epochs),
            },
            'best_epoch_metrics': {
                'loss': float(history['val_loss'][best_idx]),
                'age_loss': float(history['val_age_loss'][best_idx]),
                'gender_loss': float(history['val_gender_loss'][best_idx]),
                'age_mae': float(history['val_age_mae'][best_idx]),
                'gender_accuracy': float(history['val_gender_acc'][best_idx]),
                'gender_accuracy_percent': float(history['val_gender_acc'][best_idx] * 100),
            },
            'final_epoch_metrics': {
                'loss': float(history['val_loss'][-1]),
                'age_loss': float(history['val_age_loss'][-1]),
                'gender_loss': float(history['val_gender_loss'][-1]),
                'age_mae': float(history['val_age_mae'][-1]),
                'gender_accuracy': float(history['val_gender_acc'][-1]),
                'gender_accuracy_percent': float(history['val_gender_acc'][-1] * 100),
            },
            'improvements': {
                'age_mae_improved': history['val_age_mae'][best_idx] <= history['val_age_mae'][0],
                'gender_acc_improved': history['val_gender_acc'][best_idx] >= history['val_gender_acc'][0],
            }
        }
        
        report_path = self.log_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report_path, report


def load_model(model_class, checkpoint_path: str, device: str = 'cpu'):
    """
    Load model from checkpoint.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint
