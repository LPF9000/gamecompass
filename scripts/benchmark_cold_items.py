"""Run the metadata-only item-disjoint evidence ablation."""
from argparse import ArgumentParser
from pathlib import Path
from gamecompass.evaluation import run_cold_item_benchmark


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = run_cold_item_benchmark(args.data_dir, args.seed, args.max_users)
    print(f"Cold-item evidence ablation: {result['cold_items']} candidates; docs/benchmarks-cold-items.md")


if __name__ == '__main__':
    main()
