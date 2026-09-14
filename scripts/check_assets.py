#!/usr/bin/env python3
"""Validate the complete inference asset layout."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occluded_sitting_smplx.assets import AssetPaths, format_asset_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-detector", action="store_true")
    args = parser.parse_args()
    assets = AssetPaths.defaults(ROOT)
    print(format_asset_report(assets, include_detector=not args.no_detector))
    missing = assets.missing(include_detector=not args.no_detector)
    if missing:
        print(f"\n{len(missing)} required asset(s) missing.")
        return 1
    print("\nAll required assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
