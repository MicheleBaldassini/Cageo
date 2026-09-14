# -*- coding: utf-8 -*-
"""Run the complete SAFE-LAND workflow in dependency order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
SCRIPTS = (
    "s1_preprocessing.py",
    "s2_feature_selection.py",
    "s3_hyperparam_opt.py",
    "s4_boxplot.py",
    "s5_regplot.py",
    "s6_shap_test.py",
    "s7_shap_inference.py",
    "s8_fis.py"
)


def run_script(script_name: str) -> None:
    """Execute one scipt."""
    script_path = SRC_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage not found: {script_path}")
    print(f"\n=== Running {script_name} ===", flush=True)
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    """Execute every script sequentially."""
    for script_name in SCRIPTS:
        run_script(script_name)


if __name__ == "__main__":
    main()
