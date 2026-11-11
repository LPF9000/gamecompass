"""Download/process legitimate Steam data with explicit provenance and scope."""
from pathlib import Path
import argparse
import json
import logging
from gamecompass.ingest import acquire, acquire_kaggle, activate_kaggle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", choices=["ucsd", "kaggle"], default="kaggle")
    parser.add_argument("--user-modulus", type=int, default=32, help="Kaggle complete-user cohort; 1 means all users; default 32 development subset")
    parser.add_argument("--activate", action="store_true", help="After successful Kaggle processing, activate as primary population.sqlite and preserve UCSD research backup")
    parser.add_argument("--research-only", action="store_true", help="UCSD only: unresolved application/redistribution rights; local research only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.source == "ucsd" and args.activate:
        parser.error("--activate is supported only for the CC0 Kaggle source")
    if args.source == "ucsd":
        result = acquire(args.data_dir, research_only=args.research_only)
    else:
        def progress(scanned, selected):
            logging.info("Scanned %s source interactions; selected %s", scanned, selected)
        result = acquire_kaggle(args.data_dir, user_modulus=args.user_modulus, progress=progress)
    if args.activate:
        result["activated_database"] = str(activate_kaggle(args.data_dir))
    print(json.dumps(result, indent=2))
