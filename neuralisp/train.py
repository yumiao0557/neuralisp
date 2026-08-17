from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from neuralisp.config import TrainConfig
from neuralisp.data.datasets import FullImageTestDataset, PatchTrainDataset
from neuralisp.data.degradation import bilinear_demosaic, degrade
from neuralisp.losses import ISPLoss
from neuralisp.metrics import psnr as psnr_metric
from neuralisp.metrics import ssim_metric
from neuralisp.models.unet import JointISPNet, demosaic_denoise

ROOT = Path(__file__).resolve().parent.parent


def _save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    """tensor: (3, H, W) in [0,1] -> PNG on disk, no torchvision dependency."""
    import numpy as np
    from PIL import Image

    arr = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--patch-size", type=int, default=cfg.patch_size)
    p.add_argument("--patches-per-image", type=int, default=cfg.patches_per_image)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--train-root", type=str, default=cfg.train_root)
    p.add_argument("--val-root", type=str, default=cfg.val_root)
    p.add_argument("--checkpoint-dir", type=str, default=cfg.checkpoint_dir)
    p.add_argument("--log-dir", type=str, default=cfg.log_dir)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    p.add_argument("--limit-val", type=int, default=None, help="only evaluate first N val images (smoke tests)")
    args = p.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.patch_size = args.patch_size
    cfg.patches_per_image = args.patches_per_image
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers
    cfg.train_root = args.train_root
    cfg.val_root = args.val_root
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.log_dir = args.log_dir
    cfg._run_name = args.run_name
    cfg._resume = args.resume
    cfg._max_steps = args.max_steps
    cfg._limit_val = args.limit_val
    return cfg


@torch.no_grad()
def validate(model, val_loader, device, cfg, writer=None, epoch=0, save_samples_dir=None):
    model.eval()
    torch.manual_seed(1234)  # reproducible degradation for comparable val numbers across runs

    pred_psnrs, pred_ssims, base_psnrs = [], [], []
    for i, (clean, name) in enumerate(val_loader):
        if cfg._limit_val is not None and i >= cfg._limit_val:
            break
        clean = clean.to(device)
        out = degrade(clean, gain_range=cfg.noise_gain_range)
        baseline = bilinear_demosaic(out.packed_bayer)
        pred = demosaic_denoise(model, out.packed_bayer, out.noise_map, baseline)

        pred_psnrs.append(psnr_metric(pred, out.target_linear_rgb))
        pred_ssims.append(ssim_metric(pred, out.target_linear_rgb))
        base_psnrs.append(psnr_metric(baseline, out.target_linear_rgb))

        if save_samples_dir is not None and i < 4:
            save_samples_dir.mkdir(parents=True, exist_ok=True)
            grid = torch.cat([baseline[0], pred[0], out.target_linear_rgb[0]], dim=2)  # side by side
            _save_tensor_png(grid, save_samples_dir / f"epoch{epoch:04d}_{name[0]}.png")

    model.train()
    avg_pred_psnr = sum(pred_psnrs) / max(len(pred_psnrs), 1)
    avg_pred_ssim = sum(pred_ssims) / max(len(pred_ssims), 1)
    avg_base_psnr = sum(base_psnrs) / max(len(base_psnrs), 1)

    if writer is not None:
        writer.add_scalar("val/psnr_pred", avg_pred_psnr, epoch)
        writer.add_scalar("val/ssim_pred", avg_pred_ssim, epoch)
        writer.add_scalar("val/psnr_bilinear_baseline", avg_base_psnr, epoch)

    return avg_pred_psnr, avg_pred_ssim, avg_base_psnr


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg._run_name or time.strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir = ROOT / cfg.checkpoint_dir / run_name
    log_dir = ROOT / cfg.log_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PatchTrainDataset(
        ROOT / cfg.train_root, patch_size=cfg.patch_size, patches_per_image=cfg.patches_per_image
    )
    val_ds = FullImageTestDataset(ROOT / cfg.val_root, max_size=cfg.val_max_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = JointISPNet(
        in_channels=6,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        blocks_per_level=cfg.blocks_per_level,
        bottleneck_blocks=cfg.bottleneck_blocks,
    ).to(device)
    print(f"model params: {model.num_params() / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = ISPLoss(l1_weight=cfg.l1_weight, ssim_weight=cfg.ssim_weight)

    start_epoch = 0
    best_psnr = -1e9
    if cfg._resume:
        ckpt = torch.load(cfg._resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", best_psnr)
        print(f"resumed from {cfg._resume} at epoch {start_epoch}")

    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = start_epoch * len(train_loader)

    print(f"train patches: {len(train_ds)} | val images: {len(val_ds)} | device: {device}")

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        running = {"total": 0.0, "l1": 0.0, "ssim_loss": 0.0}
        n_batches = 0

        for clean in train_loader:
            clean = clean.to(device, non_blocking=True)
            out = degrade(clean, gain_range=cfg.noise_gain_range)
            baseline = bilinear_demosaic(out.packed_bayer)
            pred = demosaic_denoise(model, out.packed_bayer, out.noise_map, baseline)

            losses = loss_fn(pred, out.target_linear_rgb)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
                writer.add_scalar("train/loss_l1", losses["l1"].item(), global_step)
                writer.add_scalar("train/loss_ssim", losses["ssim_loss"].item(), global_step)

            if cfg._max_steps is not None and global_step >= cfg._max_steps:
                stop = True
                break

        scheduler.step()
        dt = time.time() - t0
        avg_total = running["total"] / max(n_batches, 1)
        print(f"epoch {epoch}: loss={avg_total:.4f} ({n_batches} batches, {dt:.1f}s)")

        if (epoch % cfg.val_every_epochs == 0) or stop:
            pred_psnr, pred_ssim, base_psnr = validate(
                model, val_loader, device, cfg, writer=writer, epoch=epoch,
                save_samples_dir=ROOT / "outputs" / run_name,
            )
            print(
                f"  val: pred_psnr={pred_psnr:.2f}dB pred_ssim={pred_ssim:.4f} "
                f"bilinear_baseline_psnr={base_psnr:.2f}dB"
            )

            if pred_psnr > best_psnr:
                best_psnr = pred_psnr
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_psnr": best_psnr, "cfg": cfg.__dict__},
                    ckpt_dir / "best.pt",
                )

        if (epoch % cfg.save_every_epochs == 0) or stop:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_psnr": best_psnr, "cfg": cfg.__dict__},
                ckpt_dir / "latest.pt",
            )

        if stop:
            break

    writer.close()
    print(f"done. best val psnr: {best_psnr:.2f}dB. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
