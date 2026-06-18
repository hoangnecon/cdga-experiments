#!/usr/bin/env python3
"""
Method: SAGS
Component: Training Entry Point
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.trainer import run_training
from .model import build_model

if __name__ == "__main__":
    run_training(build_model, description="Train SAGS Model")
