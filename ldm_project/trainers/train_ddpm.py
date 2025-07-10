import argparse
import yaml
import os
import logging
import itertools
import copy
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

# import dataset and model
from ldm_project.dataloader.dataset import DataSet
from ldm_project.model.ddpm import Diffusion as Model
from ldm_project.trainers.callback import (
    EarlyStopping,
    CheckpointResume,
    ModelCheckpoint,
)


# === Core Trainer ===
class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config.get("device", "cpu") if torch.cuda.is_available() else "cpu"
        )
        self.logger = self._setup_logging()
        self.start_epoch = 1
        self.stop_training = False
        self._log_hyperparameters()
        self._load_data()
        self._build_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self.callbacks = []

    def _setup_logging(self):
        log_cfg = self.config["logging"]
        os.makedirs(log_cfg["log_dir"], exist_ok=True)
        fname = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
        path = os.path.join(log_cfg["log_dir"], fname)

        logger = logging.getLogger("UniversalTrainer")
        logger.setLevel(getattr(logging, log_cfg["level"].upper()))
        # File handler
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)
        logger.info("Logging initialized.")
        return logger

    def _log_hyperparameters(self):
        self.logger.info("=== Configuration ===")
        for section, values in self.config.items():
            if isinstance(values, dict):
                self.logger.info(f"[{section}]")
                for k, v in values.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{section}: {values}")

    def _load_data(self):
        data_cfg = self.config["data"]
        train_ds = DataSet(
            root=data_cfg["train_path"],
            split="train",
        )
        val_ds = DataSet(
            root=data_cfg["val_path"],
            split="val",
        )
        t_cfg = self.config["training"]
        self.train_loader = DataLoader(
            train_ds,
            batch_size=t_cfg["batch_size"],
            shuffle=True,
            num_workers=t_cfg["num_workers"],
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=t_cfg["batch_size"],
            shuffle=False,
            num_workers=t_cfg["num_workers"],
        )
        self.logger.info("Data loaders ready.")

    def _build_model(self):
        self.model = Model(**self.config["model"]["params"]).to(self.device)
        # if parallel training is enabled
        if self.config["training"].get("parallel", False):
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self.logger.info(
                    f"Using {torch.cuda.device_count()} GPUs for training."
                )
            else:
                self.logger.warning(
                    "Parallel training enabled but only one GPU detected. "
                    "Falling back to single GPU mode."
                )
        self.logger.info(f"Model initialized on {self.device}.")

    def _setup_loss(self):
        loss_cfg = self.config.get("loss", {"type": "BCEWithLogitsLoss", "params": {}})
        if hasattr(nn, loss_cfg["type"]):
            self.criterion = getattr(nn, loss_cfg["type"])(**loss_cfg.get("params", {}))
        elif hasattr(nn.functional, loss_cfg["type"].lower()):
            self.criterion = getattr(nn.functional, loss_cfg["type"].lower())
            self.loss_params = loss_cfg.get("params", {})
        else:
            # no error, just a warning
            self.logger.warning(
                f"Loss function {loss_cfg['type']} not found in nn or nn.functional."
            )
            self.logger.warning(f"Assuming it is a custom function.")
        self.logger.info(f"Loss function ({loss_cfg['type']}) ready.")

    def _setup_optimizer(self):
        opt_cfg = self.config["optimizer"]
        self.optimizer = optim.__dict__[opt_cfg["type"]](
            self.model.parameters(), **opt_cfg["params"]
        )
        self.logger.info(f"Optimizer ({opt_cfg['type']}) ready.")

    def _setup_scheduler(self):
        sch_cfg = self.config.get("scheduler")
        if sch_cfg:
            self.scheduler = optim.lr_scheduler.__dict__[sch_cfg["type"]](
                self.optimizer, **sch_cfg["params"]
            )
            self.logger.info(f"Scheduler ({sch_cfg['type']}) ready.")
        else:
            self.scheduler = None

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(self.train_loader, 1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass
            preds, target = self.model(x)
            # Compute loss using configured criterion
            if hasattr(self.criterion, "__call__") and not isinstance(
                self.criterion, type
            ):
                # For nn.Module losses (e.g., nn.CrossEntropyLoss)
                loss = self.criterion(preds, target)
            else:
                # For functional losses (e.g., F.binary_cross_entropy)
                loss = self.criterion(preds, target, **getattr(self, "loss_params", {}))

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # Update training metrics if callback exists
            for cb in self.callbacks:
                if hasattr(cb, "on_train_batch_end"):
                    cb.on_train_batch_end(self, preds, target)

            if i % self.config["logging"]["log_interval"] == 0:
                self.logger.info(
                    f"Epoch {epoch} [{i}/{len(self.train_loader)}] Loss: {loss.item():.4f}"
                )
        return running_loss / len(self.train_loader)

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                preds, target = self.model(x)

                # Compute loss
                if hasattr(self.criterion, "__call__") and not isinstance(
                    self.criterion, type
                ):
                    # For nn.Module losses
                    loss = self.criterion(preds, target)
                else:
                    # For functional losses
                    loss = self.criterion(
                        preds, target, **getattr(self, "loss_params", {})
                    )

                total_loss += loss.item()

                # Update validation metrics if callback exists
                for cb in self.callbacks:
                    if hasattr(cb, "on_val_batch_end"):
                        cb.on_val_batch_end(self, preds, target)

        avg = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch} Validation Loss: {avg:.4f}")
        return avg

    def run(self):
        # Attach callbacks
        if self.config["training"].get("resume", False):
            self.callbacks.append(
                CheckpointResume(True, self.config["logging"]["checkpoint_dir"])
            )
        self.callbacks.append(
            EarlyStopping(self.config["training"].get("early_stopping_patience", 5))
        )
        self.callbacks.append(
            ModelCheckpoint(
                self.config["logging"]["checkpoint_dir"],
                save_every_n_epochs=self.config["logging"]["checkpoint_save_interval"],
            )
        )

        # Start training
        for cb in self.callbacks:
            cb.on_train_begin(self)

        epochs = self.config["training"]["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            logs = {"train_loss": train_loss, "val_loss": val_loss}
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self.stop_training:
                break
            if self.scheduler:
                self.scheduler.step()

        for cb in self.callbacks:
            cb.on_train_end(self)
        self.logger.info("Training complete.")


# === Hyperparameter sweep ===
def set_by_path(cfg: dict, path: str, value):
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_sweep(base_cfg: dict):
    sweep = base_cfg["sweep"]
    keys, values = zip(*sweep.items())
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)
        run_name = []
        for k, v in zip(keys, combo):
            set_by_path(cfg, k, v)
            run_name.append(f"{k.replace('.', '_')}{v}")
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_") + "_".join(run_name)
        cfg["logging"]["log_dir"] = os.path.join(cfg["logging"]["log_dir"], suffix)
        cfg["logging"]["checkpoint_dir"] = os.path.join(
            cfg["logging"]["checkpoint_dir"], suffix
        )
        Trainer(cfg).run()


# === Entry point ===
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "sweep" in config:
        run_sweep(config)
    else:
        Trainer(config).run()
