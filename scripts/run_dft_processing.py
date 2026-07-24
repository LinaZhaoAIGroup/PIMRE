#!/usr/bin/env python3
"""DFT calculation data processing pipeline.

Usage:
    uv run python scripts/run_dft_processing.py --dft-csv extracted_data.csv --band-gap-file BAND_GAP
    uv run python scripts/run_dft_processing.py --dft-csv extracted_data.csv --fermi-file FERMI_ENERGY
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pimre.pipeline.dft import run_dft_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process DFT calculation data into band map.")
    parser.add_argument("--config", default="configs/defaults.yaml", help="Path to default config")
    parser.add_argument("--dft-csv", required=True, help="Path to DFT CSV file")
    parser.add_argument("--fermi-file", default=None, help="Path to FERMI_ENERGY file")
    parser.add_argument("--band-gap-file", default=None, help="Path to BAND_GAP file")
    parser.add_argument("--output", default="band_map.h5", help="Output file path")
    parser.add_argument("--output-format", default="h5", choices=["mat", "h5"], help="Output format")
    parser.add_argument("--method", default="grid_cell", choices=["grid_cell", "griddata"],
                        help="Interpolation method")
    args = parser.parse_args()

    run_dft_pipeline(
        dft_csv=args.dft_csv,
        config_path=args.config,
        fermi_file=args.fermi_file,
        band_gap_file=args.band_gap_file,
        output=args.output,
        output_format=args.output_format,
        method=args.method,
    )


if __name__ == "__main__":
    main()