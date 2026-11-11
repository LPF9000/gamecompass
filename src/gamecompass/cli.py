"""Explicit local operations; all private state stays under the configured data directory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog='gamecompass')
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('GAMECOMPASS_DATA', 'data')))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    sub.add_parser('acquire')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--max-items', type=int, default=1500)
    bench.add_argument('--seed', type=int, default=42)
    profile = sub.add_parser('profile')
    profile.add_argument('name')
    exp = sub.add_parser('export')
    exp.add_argument('profile_id')
    exp.add_argument('destination', type=Path)
    imp = sub.add_parser('import')
    imp.add_argument('source', type=Path)
    steam = sub.add_parser('steam')
    steam.add_argument('profile_id')
    steam.add_argument('steam_id')
    app = sub.add_parser('app')
    app.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'doctor':
        import platform
        import psutil
        info = {'python':sys.version, 'platform':platform.platform(), 'cpus':os.cpu_count(),
                'available_ram_gib':round(psutil.virtual_memory().available / 2**30,2),
                'data_dir':str(args.data_dir.resolve()), 'cuda':False, 'mps':False}
        try:
            import torch
            info.update(torch=torch.__version__, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available())
        except ImportError:
            info['torch'] = 'not installed in app environment; RecBole has a separate environment'
        print(json.dumps(info, indent=2))
    elif args.command == 'benchmark':
        from .evaluation import run_benchmark
        print(json.dumps(run_benchmark(args.data_dir, max_items=args.max_items, seed=args.seed), indent=2))
    elif args.command == 'acquire':
        # Script interface remains useful from a source checkout.
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'acquire_data.py'
        subprocess.run([sys.executable,str(script),'--data-dir',str(args.data_dir.resolve())],check=True)
    elif args.command == 'app':
        env = {**os.environ, 'GAMECOMPASS_DATA':str(args.data_dir.resolve())}
        subprocess.run([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('ui.py')),
                        '--server.address','127.0.0.1','--server.port',str(args.port),
                        '--browser.gatherUsageStats','false'], env=env,check=True)
    else:
        from .store import Store
        store = Store(args.data_dir)
        if args.command == 'profile':
            print(store.create_profile(args.name))
        elif args.command == 'export':
            payload = store.export_profile(args.profile_id)
            with args.destination.open('x') as f:
                json.dump(payload,f,indent=2)
            args.destination.chmod(0o600)
            print('Exported locally. Contains personal feedback; share only intentionally.')
        elif args.command == 'import':
            print(store.import_profile(json.loads(args.source.read_text())))
        elif args.command == 'steam':
            from .integrations import SteamProvider
            games=SteamProvider().library(args.steam_id)
            store.import_library(args.profile_id,games)
            print(f'Imported {len(games)} games; playtime was not converted into ratings.')


if __name__ == '__main__':
    main()
