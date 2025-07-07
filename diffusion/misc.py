from typing import Literal
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CK_ROOT = ROOT_DIR / "log/log_2025_02_07_unconditional"


@dataclass
class ModelConfig:
    algorithm: Literal["ddpm", "ddim", "flow"] = "ddpm"
    unet_input_chan: int = 1
    unet_output_chan: int = 1
    unet_chans: int = 64
    num_pool_layers: int = 3

    # for ddpm & ddim
    ddim_eta: float = 0.0
    beta_schedule: Literal["linear", "cosine", "sigmoid"] = "cosine"
    ddpm_target: Literal["noise", "velocity", "start"] = "noise"


@dataclass
class TestConfig:
    # Dataset
    # checkpoints: str = str(CK_ROOT / "00000_train/checkpoints/checkpoint_13.ckpt")
    debugmode: bool = False

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = ROOT_DIR / "log/log_2025_02_07_unconditional_test"
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["diffusion"] = "diffusion"

    # Test params
    gpu: str = "0"
    device: torch.device | None = None
    batch_size: int = 32
    image_shape: tuple = (1, 32, 32)
    infer_len: int = 100

    # Experiment
    interval: int = 10
