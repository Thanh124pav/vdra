#!/usr/bin/env python3
"""Compatibility wrapper for the converter now owned by verl/scripts."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

VERL_ROOT = Path(__file__).resolve().parents[1] / "verl"
TARGET = VERL_ROOT / "scripts" / "convert_hf_data_to_verl_parquet.py"
os.chdir(VERL_ROOT)
runpy.run_path(str(TARGET), run_name="__main__")
