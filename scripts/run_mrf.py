#!/usr/bin/env python3
"""MRF band reconstruction pipeline with BSFI offset optimization.

Usage:
    uv run python scripts/run_mrf.py
    uv run python scripts/run_mrf.py --config configs/pimre_config.yaml
    uv run python scripts/run_mrf.py --exp-data test/exp_preprocessed.h5 --band-map test/band_map.h5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.pipeline.mrf import run_mrf_pipeline

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "configs", "pimre_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="MRF + BSFI reconstruction pipeline")
    parser.add_argument("--config", default=CONFIG_PATH, help="Config file path")
    parser.add_argument("--exp-data", default=None, help="Preprocessed exp HDF5")
    parser.add_argument("--band-map", default=None, help="DFT band map .h5")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    run_mrf_pipeline(
        config_path=args.config,
        exp_data=args.exp_data,
        band_map=args.band_map,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()