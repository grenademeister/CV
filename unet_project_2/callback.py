import os
import torch
import torch.nn.functional as F
from typing import Dict, List


# === Callback system ===
class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        pass

    def on_train_end(self, trainer):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float("inf")
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, logs):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.logger.info(
                    f"EarlyStopping: no improvement for {self.patience} epochs; stopping."
                )
                trainer.stop_training = True


class CheckpointResume(Callback):
    def __init__(self, resume: bool, checkpoint_dir: str):
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir

    def on_train_begin(self, trainer):
        if not self.resume:
            return
        if not os.path.isdir(self.checkpoint_dir):
            return
        files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("epoch_") and f.endswith(".pth")
        ]
        if not files:
            return
        epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
        last = max(epochs)
        model_path = os.path.join(self.checkpoint_dir, f"epoch_{last}.pth")
        trainer.model.load_state_dict(
            torch.load(model_path, map_location=trainer.device)
        )
        # load optimizer state
        opt_path = os.path.join(self.checkpoint_dir, f"opt_{last}.pth")
        if os.path.exists(opt_path):
            trainer.optimizer.load_state_dict(
                torch.load(opt_path, map_location=trainer.device)
            )
        # load scheduler state
        if trainer.scheduler:
            sch_path = os.path.join(self.checkpoint_dir, f"sch_{last}.pth")
            if os.path.exists(sch_path):
                trainer.scheduler.load_state_dict(
                    torch.load(sch_path, map_location=trainer.device)
                )
        trainer.start_epoch = last + 1
        trainer.logger.info(f"Resumed from epoch {last}.")


class ModelCheckpoint(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_optimizer: bool = False,
        save_scheduler: bool = False,
        save_every_n_epochs: int = 1,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode

        # Best model tracking
        if self.save_best:
            self.best_value = float("inf") if mode == "min" else float("-inf")
            self.best_epoch = 0

    def on_epoch_end(self, trainer, epoch, logs):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Save every N epochs
        if (epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, epoch, prefix="epoch")
            trainer.logger.info(f"Regular checkpoint saved: epoch {epoch}")

        # Save best model
        if self.save_best:
            current_value = logs.get(self.monitor)
            if current_value is not None:
                is_best = self._is_better(current_value, self.best_value)
                if is_best:
                    self.best_value = current_value
                    self.best_epoch = epoch
                    self._save_checkpoint(trainer, epoch, prefix="best")
                    trainer.logger.info(
                        f"Best model saved: epoch {epoch}, {self.monitor}={current_value:.4f}"
                    )

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < best
        else:  # mode == 'max'
            return current > best

    def _save_checkpoint(self, trainer, epoch, prefix="epoch"):
        """Save model, optimizer, and scheduler state"""
        # Save model
        torch.save(
            trainer.model.state_dict(),
            os.path.join(self.checkpoint_dir, f"{prefix}_{epoch}.pth"),
        )

        # Save optimizer
        if self.save_optimizer:
            torch.save(
                trainer.optimizer.state_dict(),
                os.path.join(self.checkpoint_dir, f"{prefix}_opt_{epoch}.pth"),
            )

        # Save scheduler
        if self.save_scheduler:
            torch.save(
                trainer.scheduler.state_dict(),
                os.path.join(self.checkpoint_dir, f"{prefix}_sch_{epoch}.pth"),
            )


class SegmentationMetrics:
    """
    Segmentation metrics calculator for multi-class segmentation.
    Computes pixel accuracy, mean IoU, and per-class IoU.
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all accumulated metrics"""
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions and targets

        Args:
            preds: Model predictions [B, C, H, W] (logits)
            targets: Ground truth [B, H, W] (class indices) or [B, C, H, W] (one-hot)
        """
        # Convert predictions to class indices
        if preds.dim() == 4:  # [B, C, H, W]
            pred_classes = torch.argmax(preds, dim=1)  # [B, H, W]
        else:
            pred_classes = preds

        # Convert targets to class indices if one-hot
        if targets.dim() == 4:  # [B, C, H, W] one-hot
            target_classes = torch.argmax(targets, dim=1)  # [B, H, W]
        else:
            target_classes = targets

        # Flatten for easier computation
        pred_flat = pred_classes.flatten()
        target_flat = target_classes.flatten()

        # Update pixel accuracy
        correct = (pred_flat == target_flat).sum().item()
        total = pred_flat.numel()
        self.correct_pixels += correct
        self.total_pixels += total

        # FIXED: Replace slow loop with vectorized confusion matrix update
        # Create mask for valid predictions (within class range)
        valid_mask = (
            (target_flat >= 0)
            & (target_flat < self.num_classes)
            & (pred_flat >= 0)
            & (pred_flat < self.num_classes)
        )

        valid_targets = target_flat[valid_mask]
        valid_preds = pred_flat[valid_mask]

        # Vectorized confusion matrix update using bincount
        if len(valid_targets) > 0:
            indices = valid_targets * self.num_classes + valid_preds
            cm_flat = torch.bincount(indices, minlength=self.num_classes**2)
            cm_update = cm_flat.view(self.num_classes, self.num_classes).float()
            self.confusion_matrix += cm_update

    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}

        # Pixel Accuracy
        pixel_acc = (
            (self.correct_pixels / self.total_pixels) * 100
            if self.total_pixels > 0
            else 0.0
        )
        metrics["pixel_accuracy"] = pixel_acc

        # Per-class IoU and mean IoU
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i].item()
            fp = self.confusion_matrix[:, i].sum().item() - tp
            fn = self.confusion_matrix[i, :].sum().item() - tp

            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0

            ious.append(iou * 100)
            metrics[f"iou_{self.class_names[i]}"] = iou * 100

        metrics["mean_iou"] = sum(ious) / len(ious) if ious else 0.0

        # Per-class accuracy
        for i in range(self.num_classes):
            class_total = self.confusion_matrix[i, :].sum().item()
            class_correct = self.confusion_matrix[i, i].item()
            class_acc = (class_correct / class_total) * 100 if class_total > 0 else 0.0
            metrics[f"acc_{self.class_names[i]}"] = class_acc

        return metrics

    def get_summary_string(self) -> str:
        """Get a formatted string summary of metrics"""
        metrics = self.compute()

        summary = []
        summary.append(f"Pixel Accuracy: {metrics['pixel_accuracy']:.2f}%")
        summary.append(f"Mean IoU: {metrics['mean_iou']:.2f}%")

        for i, class_name in enumerate(self.class_names):
            iou = metrics[f"iou_{class_name}"]
            acc = metrics[f"acc_{class_name}"]
            summary.append(f"{class_name}: IoU={iou:.2f}%, Acc={acc:.2f}%")

        return " | ".join(summary)


class MetricsCallback(Callback):
    """
    Callback to compute and log segmentation metrics during training
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["Background", "Unknown", "Foreground"]
        self.train_metrics = SegmentationMetrics(num_classes, class_names)
        self.val_metrics = SegmentationMetrics(num_classes, class_names)

    def on_epoch_begin(self, trainer, epoch):
        """Reset metrics at the beginning of each epoch"""
        self.train_metrics.reset()
        self.val_metrics.reset()

    def on_train_batch_end(self, trainer, preds, targets):
        """Update training metrics after each batch"""
        self.train_metrics.update(preds.detach().cpu(), targets.detach().cpu())

    def on_val_batch_end(self, trainer, preds, targets):
        """Update validation metrics after each batch"""
        self.val_metrics.update(preds.detach().cpu(), targets.detach().cpu())

    def on_epoch_end(self, trainer, epoch, logs):
        """Log metrics at the end of each epoch"""
        train_summary = self.train_metrics.get_summary_string()
        val_summary = self.val_metrics.get_summary_string()

        trainer.logger.info(f"Epoch {epoch} Training Metrics: {train_summary}")
        trainer.logger.info(f"Epoch {epoch} Validation Metrics: {val_summary}")

        # Add key metrics to logs for other callbacks (like early stopping)
        val_metrics = self.val_metrics.compute()
        logs.update(
            {
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                "val_mean_iou": val_metrics["mean_iou"],
            }
        )
