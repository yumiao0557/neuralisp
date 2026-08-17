from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # data
    train_root: str = "data_raw/train/bsds500"
    val_root: str = "data_raw/test/kodak"
    patch_size: int = 128
    patches_per_image: int = 8
    val_max_size: int = 256
    num_workers: int = 4

    # model
    base_channels: int = 32
    channel_mults: tuple[int, ...] = (1, 2, 4, 8)
    blocks_per_level: int = 2
    bottleneck_blocks: int = 4

    # noise / degradation
    noise_gain_range: tuple[float, float] = (-4.0, -1.0)  # log10(shot_a) range

    # optimization
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 40
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    l1_weight: float = 1.0
    ssim_weight: float = 0.15

    # bookkeeping
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    val_every_epochs: int = 1
    save_every_epochs: int = 1
    seed: int = 0
