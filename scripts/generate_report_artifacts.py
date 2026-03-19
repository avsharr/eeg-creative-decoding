from __future__ import annotations

import sys

from src.reporting import generate_report_artifacts


def main():
    out = generate_report_artifacts()
    print("Loaded experiments:", len(out["loaded_experiments"]))
    if out["missing_experiments"]:
        print("Missing experiments:")
        for exp in out["missing_experiments"]:
            print(" -", exp)
    print("Saved summary:", out["report_all_path"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)