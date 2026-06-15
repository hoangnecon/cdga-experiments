"""
Method: Baseline CE
Component: Training Script
Dataset: ISPRS Vaihingen / Potsdam

Ref:
    - rules/STRUCTURE.md
    - rules/CONVENTIONS.md
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.trainer import run_training
from experiments.methods.group_a.loss_based.baseline_ce.model import build_model

if __name__ == "__main__":
    run_training(build_model, description="Train Baseline CE Model")
