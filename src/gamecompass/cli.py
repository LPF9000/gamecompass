"""Explicit local operations; all private state stays under the configured data directory."""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


def _execute() -> None:
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(prog='gamecompass')
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('GAMECOMPASS_DATA', 'data')))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    learn = sub.add_parser('feedback-snapshot')
    learn.add_argument('output_dir',type=Path)
    acquire = sub.add_parser('acquire')
    acquire.add_argument('--source', choices=['kaggle','ucsd'], default='kaggle')
    acquire.add_argument('--research-only',action='store_true')
    acquire.add_argument('--user-modulus',type=int,default=32)
    acquire.add_argument('--activate',action='store_true')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--max-items', type=int, default=1500)
    bench.add_argument('--seed', type=int, default=42)
    bench.add_argument('--max-users',type=int,default=300)
    temporal = sub.add_parser('temporal-benchmark')
    temporal.add_argument('--max-items',type=int,default=1500)
    activate = sub.add_parser('activate-model')
    activate.add_argument('version')
    personal = sub.add_parser('personal-test')
    personal.add_argument('profile_id')
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
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
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
    elif args.command == 'feedback-snapshot':
        from .learning import incorporate_feedback
        print(json.dumps(incorporate_feedback(args.data_dir,args.output_dir),indent=2))
    elif args.command == 'benchmark':
        from .evaluation import run_benchmark, run_implicit_benchmark
        report = run_benchmark(args.data_dir, max_items=args.max_items, seed=args.seed,max_users=args.max_users)
        implicit = run_implicit_benchmark(args.data_dir,seed=args.seed,max_users=args.max_users)
        print(json.dumps({'benchmark':report,'implicit_adapter':implicit},indent=2))
    elif args.command == 'temporal-benchmark':
        from .evaluation import run_temporal_benchmark
        print(json.dumps(run_temporal_benchmark(args.data_dir,max_items=args.max_items),indent=2))
    elif args.command == 'activate-model':
        from .registry import activate
        activate(args.data_dir,args.version)
        print('Activated model after integrity verification')
    elif args.command == 'personal-test':
        from .personal import evaluate_profile
        print(json.dumps(evaluate_profile(args.data_dir,args.profile_id),indent=2))
    elif args.command == 'acquire':
        # Script interface remains useful from a source checkout.
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'acquire_data.py'
        command = [sys.executable,str(script),'--data-dir',str(args.data_dir.resolve()),'--source',args.source,'--user-modulus',str(args.user_modulus)]
        if args.research_only: command.append('--research-only')
        if args.activate: command.append('--activate')
        subprocess.run(command,check=True)
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


def main() -> None:
    try:
        _execute()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'GameCompass: {exc}',file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
