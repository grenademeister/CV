import os
import torch
import torch.nn.functional as F
from typing import Dict, List
import matplotlib.pyplot as plt
import torchvision
import numpy as np


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
            if f.startswith("epoch_")
            and f.endswith(".pth")
            and not ("_opt_" in f or "_sch_" in f)
        ]
        if not files:
            return
        epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
        last = max(epochs)
        model_path = os.path.join(self.checkpoint_dir, f"epoch_{last}.pth")
        trainer.model.load_state_dict(
            torch.load(model_path, map_location=trainer.device)
        )
        trainer.logger.info(f"Model loaded from {model_path}")
        # load optimizer state
        opt_path = os.path.join(self.checkpoint_dir, f"epoch_opt_{last}.pth")
        if os.path.exists(opt_path):
            trainer.optimizer.load_state_dict(
                torch.load(opt_path, map_location=trainer.device)
            )
            trainer.logger.info(f"Optimizer loaded from {opt_path}")
        # load scheduler state
        if trainer.scheduler:
            sch_path = os.path.join(self.checkpoint_dir, f"epoch_sch_{last}.pth")
            if os.path.exists(sch_path):
                trainer.scheduler.load_state_dict(
                    torch.load(sch_path, map_location=trainer.device)
                )
                trainer.logger.info(f"Scheduler loaded from {sch_path}")
        trainer.start_epoch = last + 1
        trainer.logger.info(f"Resumed from epoch {last}.")


class ModelCheckpoint(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
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
            os.path.join(
                self.checkpoint_dir,
                f"{prefix}_{epoch}.pth" if prefix == "epoch" else f"{prefix}.pth",
            ),
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
