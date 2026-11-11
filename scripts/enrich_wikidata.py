"""Acquire free CC0 source-entity platform/date facts, without edition merging."""
import argparse
import json
import logging
from pathlib import Path

from gamecompass.wikidata import acquire_enrichment, enrich_parent_work_dates

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--parent-work-dates", action="store_true", help="Optional bounded one-hop explicit P629 parent-work date follow-up")
    p.add_argument("--apply", action="store_true", help="Add separate wikidata:Q entries to local catalog; leave Steam records/population untouched")
    args = p.parse_args()
    current = json.loads((args.data_dir / "models/current.json").read_text())["version"]
    model = json.loads((args.data_dir / "models" / current / "manifest.json").read_text())
    logging.basicConfig(level=logging.INFO)
    result = enrich_parent_work_dates(args.data_dir, apply=args.apply) if args.parent_work_dates else acquire_enrichment(args.data_dir,model["cf_game_ids"][:args.limit],apply=args.apply)
    print(json.dumps(result,indent=2))
