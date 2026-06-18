"""
Component: Shared Training Runner
Location: shared/trainer.py

Purpose:
- Handles standard training and validation loops
- Coordinates argument parsing and config loading/merging/override
- Captures environment metadata for reproducibility
- Implements checkpoint saving and loading
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import yaml

# Set project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # SAGM/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.metrics.region import compute_miou, compute_mf1, compute_oa
from shared.metrics.boundary import compute_boundary_f1
from shared.utils.logging_utils import ExperimentLogger
from shared.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from shared.utils.seed_utils import set_seed


# ══════════════════════════════════════════════════════════════
# 1. Argument Parsing
# ══════════════════════════════════════════════════════════════

def parse_args(description: str = "Train Model") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--env-config", type=Path, default=None,
        help="Path to YAML environment config file (for overriding paths).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override auto-generated run ID.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from (last.pth).",
    )
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values. Example: --override cdga.gamma=20 train.lr=1e-4",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse config and setup, but do NOT train.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# 2. Config Loading and Validation
# ══════════════════════════════════════════════════════════════

def load_config(config_path: Path, env_config_path: Optional[Path] = None) -> dict:
    """Load YAML config, merging base config if specified, then merge env config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Merge base config if specified
    if "base" in cfg:
        base_val = cfg.pop("base")
        # Try to resolve relative to method config folder
        base_path = config_path.parent / base_val
        if not base_path.exists():
            # Try to resolve relative to configs folder
            base_path = PROJECT_ROOT / "experiments" / "configs" / "base" / base_val
        
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    # 2. Merge environment config (env config overrides method config)
    if env_config_path is not None and env_config_path.exists():
        with open(env_config_path) as f:
            env_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, env_cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins on conflict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply KEY=VALUE string overrides to nested config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}  # Create nested dict if not present
            d = d[k]
        # Auto-cast: try int, then float, then bool, then string
        try:
            d[keys[-1]] = int(value)
        except ValueError:
            try:
                d[keys[-1]] = float(value)
            except ValueError:
                if value.lower() == "true":
                    d[keys[-1]] = True
                elif value.lower() == "false":
                    d[keys[-1]] = False
                else:
                    d[keys[-1]] = value
    return cfg


def validate_config(cfg: dict) -> None:
    """Validate required config fields are present."""
    required_sections = ["project", "data", "model", "train", "eval"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")
    if cfg["project"].get("seed") is None:
        raise ValueError("project.seed must be set. Reproducibility is mandatory.")


# ══════════════════════════════════════════════════════════════
# 3. Run Directory Setup
# ══════════════════════════════════════════════════════════════

def setup_run_dir(cfg: dict, run_id: Optional[str] = None) -> Path:
    """Create and initialize the run directory."""
    if run_id is None:
        method = cfg["project"]["method"]
        backbone = cfg["model"]["backbone_slug"]
        dataset = cfg["data"]["dataset"]
        tz = timezone(timedelta(hours=7))
        ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        run_id = f"{method}__{backbone}__{dataset}__{ts}"

    # Use runs_dir from env config if set, otherwise default to local
    runs_parent = cfg.get("output", {}).get("runs_dir")
    if runs_parent:
        runs_dir = Path(runs_parent)
    else:
        runs_dir = PROJECT_ROOT / "experiments" / "runs"
        
    run_dir = runs_dir / run_id

    # Create subdirectories
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    return run_dir


def freeze_run_metadata(cfg: dict, run_dir: Path) -> None:
    """Save frozen config and environment metadata to run directory."""
    # 1. Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        git_hash = "N/A"
        git_branch = "N/A"

    (run_dir / "git_hash.txt").write_text(f"{git_branch}\n{git_hash}\n")

    # 2. pip freeze
    try:
        pip_output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).decode()
        (run_dir / "env.txt").write_text(pip_output)
    except Exception:
        (run_dir / "env.txt").write_text("Could not capture pip freeze.\n")

    # 3. GPU info
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # 4. Add runtime section to config and freeze it
    tz = timezone(timedelta(hours=7))
    cfg["_runtime"] = {
        "run_id": run_dir.name,
        "started_at": datetime.now(tz).isoformat(),
        "git_commit": git_hash,
        "git_branch": git_branch,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "local",
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


# ══════════════════════════════════════════════════════════════
# 4. Training Loop Functions
# ══════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    cfg: dict,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train_mode()
    total_loss = 0.0
    t0 = time.time()

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch.get("boundary_mask")
        if masks is not None:
            masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["train"].get("amp", True)):
            loss = model.forward_train(images, labels, boundary_mask=masks)

        scaler.scale(loss).backward()

        if cfg["train"].get("grad_clip") is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["grad_clip"]
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "time_s": time.time() - t0,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    """Run validation and compute metrics on [0, 100] scale."""
    model.eval_mode()
    t0 = time.time()

    all_preds, all_targets = [], []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        preds = model.forward_inference(images)
        preds = preds.cpu()

        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    num_classes = cfg["data"]["num_classes"]
    ignore_index = cfg["data"].get("ignore_index", 255)
    
    region_metrics = compute_miou(all_preds, all_targets, num_classes, ignore_index)
    mf1 = compute_mf1(all_preds, all_targets, num_classes, ignore_index)
    oa = compute_oa(all_preds, all_targets, ignore_index)
    boundary_metrics = compute_boundary_f1(
        all_preds, all_targets,
        dilation_widths=cfg["eval"].get("boundary_widths", [3, 5]),
        ignore_index=ignore_index,
    )

    return {
        "miou": region_metrics["miou"] * 100,
        "mf1": mf1 * 100,
        "oa": oa * 100,
        "bf1_3": boundary_metrics["bf1_3"] * 100,
        "bf1_5": boundary_metrics["bf1_5"] * 100,
        "per_class_iou": region_metrics["per_class_iou"],
        "time_s": time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════
# 5. Dynamic Dataset Loading
# ══════════════════════════════════════════════════════════════

def get_dataset_class(dataset_name: str) -> type:
    """Import and return dataset class dynamically."""
    dataset_name = dataset_name.lower().replace("-", "_")
    if dataset_name == "vaihingen":
        from shared.datasets.vaihingen import VaihingenDataset
        return VaihingenDataset
    elif dataset_name in ["potsdam", "potsdamrgb"]:
        from shared.datasets.potsdam import PotsdamDataset
        return PotsdamDataset
    elif dataset_name == "loveda":
        from shared.datasets.loveda import LoveDADataset
        return LoveDADataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# ══════════════════════════════════════════════════════════════
# 6. Global Runner Function
# ══════════════════════════════════════════════════════════════

def run_training(build_model_fn: Callable[[dict], nn.Module], description: str = "Training Script") -> None:
    """Global generic execution entry point for all method scripts."""
    args = parse_args(description)

    # Load and merge configs
    cfg = load_config(args.config, args.env_config)
    cfg = apply_overrides(cfg, args.override)
    validate_config(cfg)

    # Set seed
    set_seed(cfg["project"]["seed"])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run dir setup
    run_dir = setup_run_dir(cfg, run_id=args.run_id)
    freeze_run_metadata(cfg, run_dir)

    # Logger setup
    logger = ExperimentLogger(run_dir=run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device} ({cfg['_runtime']['gpu']})")
    logger.info(f"Seed: {cfg['project']['seed']}")
    
    # Log formatted configuration table
    logger.log_config(cfg)


    if args.dry_run:
        logger.info("[DRY RUN] Setup validated successfully. Exiting.")
        logger.close()
        return

    # Load dataset classes
    dataset_cls = get_dataset_class(cfg["data"]["dataset"])
    
    # Initialize datasets
    train_dataset = dataset_cls(
        split="train",
        crop_size=cfg["data"]["crop_size"],
        data_root=cfg["data"]["data_root"],
    )
    val_dataset = dataset_cls(
        split="val",
        crop_size=None,  # Evaluate on full patches
        data_root=cfg["data"]["data_root"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    logger.info(f"Dataset: {cfg['data']['dataset']} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Build model wrapper
    model = build_model_fn(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model backbone: {cfg['model']['backbone']} | Parameters: {n_params:,}")

    # Optimizer & LR scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True))

    # Resume capability
    start_epoch = 1
    metrics_history = []
    best_miou = 0.0
    best_bf1 = 0.0

    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        best_bf1 = ckpt.get("best_bf1", 0.0)
        metrics_history = ckpt.get("metrics_history", [])
        logger.info(f"Resumed from epoch {ckpt['epoch']} | best mIoU: {best_miou:.2f} | best BF1@3: {best_bf1:.2f}")

    # Training Loop
    ckpt_dir = run_dir / "checkpoints"
    total_epochs = cfg["train"]["epochs"]

    for epoch in range(start_epoch, total_epochs + 1):
        # Sync epoch to model for phase-based methods (e.g., ABL activates at epoch 80)
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, cfg
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        logger.log_epoch(epoch=epoch, phase="train",
                         loss=train_metrics["loss"],
                         lr=current_lr,
                         time_s=train_metrics["time_s"])

        val_metrics = validate(model, val_loader, device, cfg)
        logger.log_epoch(epoch=epoch, phase="val", **val_metrics)

        # Log hook stats if available
        if hasattr(model, "get_hook_stats"):
            stats = model.get_hook_stats()
            if stats:
                logger.log_hook_stats(epoch=epoch, stats=stats)

        # Track history
        metrics_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        _append_epoch_csv(run_dir / "metrics" / "per_epoch.csv", epoch, val_metrics)

        # Checkpoint evaluation
        is_best_miou = val_metrics["miou"] > best_miou
        is_best_bf1 = val_metrics["bf1_3"] > best_bf1

        if is_best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(ckpt_dir / "best_miou.pth", epoch, model,
                            optimizer, scheduler, val_metrics, cfg, metrics_history)
            logger.log_best_event(epoch, "mIoU", best_miou, best_miou)

        if is_best_bf1:
            best_bf1 = val_metrics["bf1_3"]
            save_checkpoint(ckpt_dir / "best_bf1.pth", epoch, model,
                            optimizer, scheduler, val_metrics, cfg, metrics_history)
            logger.log_best_event(epoch, "BF1@3", best_bf1, best_bf1)

        # Save last checkpoint
        save_checkpoint(ckpt_dir / "last.pth", epoch, model,
                        optimizer, scheduler, val_metrics, cfg, metrics_history)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pth", epoch, model,
                            optimizer, scheduler, val_metrics, cfg, metrics_history)

    # Save final summary metrics
    final_metrics = {
        "best_miou": best_miou,
        "best_bf1_3": best_bf1,
        "total_epochs": total_epochs,
        "run_id": run_dir.name,
    }
    with open(run_dir / "metrics" / "val_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.log_final_metrics(**final_metrics)
    logger.close()


def run_evaluation(build_model_fn: Callable[[dict], nn.Module], description: str = "Evaluation Script") -> None:
    """Global generic evaluation entry point for all method scripts."""
    args = parse_args(description)

    # Load and merge configs
    cfg = load_config(args.config, args.env_config)
    cfg = apply_overrides(cfg, args.override)
    validate_config(cfg)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model_fn(cfg).to(device)

    # Load weights
    if args.resume is None:
        raise ValueError("Please specify the checkpoint to evaluate using --resume")
    ckpt = load_checkpoint(args.resume, model)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load dataset class
    dataset_cls = get_dataset_class(cfg["data"]["dataset"])
    val_dataset = dataset_cls(
        split="val",
        crop_size=None,
        data_root=cfg["data"]["data_root"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Validate
    val_metrics = validate(model, val_loader, device, cfg)
    print("==========================================")
    print(f"Evaluation Results for {cfg['project']['name']}:")
    print(f"  mIoU:   {val_metrics['miou']:.2f}%")
    print(f"  mF1:    {val_metrics['mf1']:.2f}%")
    print(f"  OA:     {val_metrics['oa']:.2f}%")
    print(f"  BF1@3:  {val_metrics['bf1_3']:.2f}%")
    print(f"  BF1@5:  {val_metrics['bf1_5']:.2f}%")
    print("==========================================")


def _append_epoch_csv(csv_path: Path, epoch: int, metrics: dict) -> None:
    header = "epoch,miou,mf1,oa,bf1_3,bf1_5\n"
    row = (f"{epoch},{metrics['miou']:.4f},{metrics['mf1']:.4f},"
           f"{metrics['oa']:.4f},{metrics['bf1_3']:.4f},{metrics['bf1_5']:.4f}\n")
    if not csv_path.exists():
        csv_path.write_text(header + row)
    else:
        with open(csv_path, "a") as f:
            f.write(row)

