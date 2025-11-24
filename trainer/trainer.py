"""Trainer class for Age-Gender prediction model."""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from .metrics import get_gender_accuracy, get_age_mae, get_age_rmse


class Trainer:
    """
    Trainer class to manage the training and validation loop.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        alpha: float = 1.0,
        beta: float = 1.0,
        logger=None,
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            criterion: Loss function (MultiTaskLoss).
            optimizer: Optimizer.
            device: Device to use (cpu/cuda).
            checkpoint_dir: Directory to save checkpoints.
            alpha: Weight for age loss.
            beta: Weight for gender loss.
            logger: Logger instance for logging.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.alpha = alpha
        self.beta = beta
        self.logger = logger
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking metrics
        self.train_history = {
            "epoch": [],
            "train_loss": [],
            "train_age_loss": [],
            "train_gender_loss": [],
            "train_gender_acc": [],
            "train_age_mae": [],
            "val_loss": [],
            "val_age_loss": [],
            "val_gender_loss": [],
            "val_gender_acc": [],
            "val_age_mae": [],
        }
        
        self.best_val_loss = float("inf")
        self.best_epoch = 0
    
    def log(self, message):
        """Log message to logger or print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary with training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_age_loss = 0.0
        total_gender_loss = 0.0
        all_gender_preds = []
        all_gender_labels = []
        all_age_preds = []
        all_age_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (images, sample) in enumerate(pbar):
            images = images.to(self.device)
            ages = sample['age'].to(self.device)
            genders = sample['gender'].to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            age_pred, gender_pred = self.model(images)
            
            # Compute loss (returns tuple: total_loss, loss_age, loss_gender)
            loss, loss_age, loss_gender = self.criterion(age_pred, gender_pred, ages, genders)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_age_loss += loss_age.item()
            total_gender_loss += loss_gender.item()
            all_gender_preds.append(gender_pred.detach().cpu())
            all_gender_labels.append(genders.detach().cpu())
            all_age_preds.append(age_pred.detach().cpu())
            all_age_labels.append(ages.detach().cpu())
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Age Loss': f"{loss_age.item():.4f}",
                'Gender Loss': f"{loss_gender.item():.4f}"
            })
        
        # Compute epoch metrics
        all_gender_preds = torch.cat(all_gender_preds, dim=0)
        all_gender_labels = torch.cat(all_gender_labels, dim=0)
        all_age_preds = torch.cat(all_age_preds, dim=0)
        all_age_labels = torch.cat(all_age_labels, dim=0)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_age_loss = total_age_loss / len(self.train_loader)
        avg_gender_loss = total_gender_loss / len(self.train_loader)
        gender_acc = get_gender_accuracy(all_gender_preds.squeeze(), all_gender_labels.squeeze())
        age_mae = get_age_mae(all_age_preds.squeeze(), all_age_labels.squeeze())
        
        return {
            "loss": avg_loss,
            "age_loss": avg_age_loss,
            "gender_loss": avg_gender_loss,
            "gender_acc": gender_acc,
            "age_mae": age_mae,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_age_loss = 0.0
        total_gender_loss = 0.0
        all_gender_preds = []
        all_gender_labels = []
        all_age_preds = []
        all_age_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, sample in pbar:
                images = images.to(self.device)
                ages = sample['age'].to(self.device)
                genders = sample['gender'].to(self.device).float()
                
                # Forward pass
                age_pred, gender_pred = self.model(images)
                
                # Compute loss
                loss, loss_age, loss_gender = self.criterion(age_pred, gender_pred, ages, genders)
                
                # Track metrics
                total_loss += loss.item()
                total_age_loss += loss_age.item()
                total_gender_loss += loss_gender.item()
                all_gender_preds.append(gender_pred.detach().cpu())
                all_gender_labels.append(genders.detach().cpu())
                all_age_preds.append(age_pred.detach().cpu())
                all_age_labels.append(ages.detach().cpu())
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Age Loss': f"{loss_age.item():.4f}",
                    'Gender Loss': f"{loss_gender.item():.4f}"
                })
        
        # Compute epoch metrics
        all_gender_preds = torch.cat(all_gender_preds, dim=0)
        all_gender_labels = torch.cat(all_gender_labels, dim=0)
        all_age_preds = torch.cat(all_age_preds, dim=0)
        all_age_labels = torch.cat(all_age_labels, dim=0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_age_loss = total_age_loss / len(self.val_loader)
        avg_gender_loss = total_gender_loss / len(self.val_loader)
        gender_acc = get_gender_accuracy(all_gender_preds.squeeze(), all_gender_labels.squeeze())
        age_mae = get_age_mae(all_age_preds.squeeze(), all_age_labels.squeeze())
        
        return {
            "loss": avg_loss,
            "age_loss": avg_age_loss,
            "gender_loss": avg_gender_loss,
            "gender_acc": gender_acc,
            "age_mae": age_mae,
        }
    
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, best_path)
            self.log(f"✓ Saved best model at epoch {epoch}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.train_history["epoch"]
        
        # Plot total loss
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'o-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 's-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot component losses
        axes[0, 1].plot(epochs, self.train_history['train_age_loss'], 'o-', label='Train Age Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.train_history['val_age_loss'], 's-', label='Val Age Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.train_history['train_gender_loss'], '^-', label='Train Gender Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.train_history['val_gender_loss'], 'v-', label='Val Gender Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Component Losses (α={}, β={})'.format(self.alpha, self.beta))
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot age MAE
        axes[1, 0].plot(epochs, self.train_history['train_age_mae'], 'o-', label='Train Age MAE', linewidth=2)
        axes[1, 0].plot(epochs, self.train_history['val_age_mae'], 's-', label='Val Age MAE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE (years)')
        axes[1, 0].set_title('Age Mean Absolute Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot gender accuracy
        train_gender_acc = [acc * 100 for acc in self.train_history['train_gender_acc']]
        val_gender_acc = [acc * 100 for acc in self.train_history['val_gender_acc']]
        axes[1, 1].plot(epochs, train_gender_acc, 'o-', label='Train Gender Acc', linewidth=2)
        axes[1, 1].plot(epochs, val_gender_acc, 's-', label='Val Gender Acc', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Gender Classification Accuracy')
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / "training_history.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        self.log(f"✓ Plot saved: {plot_path}")
        plt.close()
    
    def fit(self, epochs: int, patience: int = 10, saver=None):
        """
        Fit the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train.
            patience: Early stopping patience (number of epochs without improvement).
            saver: ResultsSaver instance for saving results.
        """
        self.log(f"\n📈 Training Configuration:")
        self.log(f"   • Total Epochs: {epochs}")
        self.log(f"   • Early Stopping Patience: {patience}")
        self.log(f"   • Device: {self.device}")
        self.log(f"   • Training Samples: {len(self.train_loader.dataset)}")
        self.log(f"   • Validation Samples: {len(self.val_loader.dataset)}\n")
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch [{epoch}/{epochs}]")
            self.log(f"{'='*80}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate()
            
            # Update history
            self.train_history["epoch"].append(epoch)
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["train_age_loss"].append(train_metrics["age_loss"])
            self.train_history["train_gender_loss"].append(train_metrics["gender_loss"])
            self.train_history["train_gender_acc"].append(train_metrics["gender_acc"])
            self.train_history["train_age_mae"].append(train_metrics["age_mae"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["val_age_loss"].append(val_metrics["age_loss"])
            self.train_history["val_gender_loss"].append(val_metrics["gender_loss"])
            self.train_history["val_gender_acc"].append(val_metrics["gender_acc"])
            self.train_history["val_age_mae"].append(val_metrics["age_mae"])
            
            # Print metrics
            self.log(f"\n📊 Train Metrics:")
            self.log(f"   • Total Loss: {train_metrics['loss']:.4f}")
            self.log(f"     ├─ Age Loss: {train_metrics['age_loss']:.4f}")
            self.log(f"     └─ Gender Loss: {train_metrics['gender_loss']:.4f}")
            self.log(f"   • Age MAE: {train_metrics['age_mae']:.2f} years")
            self.log(f"   • Gender Accuracy: {train_metrics['gender_acc']*100:.2f}%")
            
            self.log(f"\n✅ Validation Metrics:")
            self.log(f"   • Total Loss: {val_metrics['loss']:.4f}")
            self.log(f"     ├─ Age Loss: {val_metrics['age_loss']:.4f}")
            self.log(f"     └─ Gender Loss: {val_metrics['gender_loss']:.4f}")
            self.log(f"   • Age MAE: {val_metrics['age_mae']:.2f} years")
            self.log(f"   • Gender Accuracy: {val_metrics['gender_acc']*100:.2f}%")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                self.log(f"\n⭐ New best validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                self.log(f"\n⚠️  No improvement for {patience_counter}/{patience} epochs")
            
            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                
                # Save with ResultsSaver if provided
                if saver:
                    saver.save_model(self.model, self.optimizer, epoch, val_metrics, is_best=True)
                
                self.log(f"\n⭐ New best validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                self.log(f"\n⚠️  No improvement for {patience_counter}/{patience} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                self.log(f"\n🛑 Early stopping triggered!")
                break
        
        self.log(f"\n{'='*80}")
        self.log(f"✓ Training completed!")
        self.log(f"Best model at epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")
        self.log(f"{'='*80}\n")
        
        # Save with ResultsSaver if provided
        if saver:
            self.log(f"💾 Saving training results...")
            saver.save_training_history(self.train_history)
            saver.plot_training_curves(self.train_history)
            saver.plot_epoch_comparison(self.train_history, self.best_epoch)
            
            config = {
                'max_epochs': epochs,
                'patience': patience,
                'optimizer': 'Adam',
            }
            saver.create_training_report(self.train_history, self.best_epoch, epoch, config)
            self.log(f"   ✓ Results saved to: {saver.checkpoint_dir}\n")
        
        # Plot training history
        self.log(f"📉 Plotting training history...")
        self.plot_training_history()
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=4)
        self.log(f"✓ Training history saved to {history_path}\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
