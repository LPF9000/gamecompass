"""Package a trained catalog-only runtime without population histories or local profiles."""
from pathlib import Path
import argparse
import json
import sqlite3
import tempfile
import zipfile
from gamecompass.models import Model


def package_runtime(data_dir: Path, destination: Path, catalog_source: Path | None = None) -> dict:
    pointer=json.loads((data_dir/'models'/'current.json').read_text())
    version=pointer['version']
    if Path(version).name != version:
        raise ValueError('Invalid model version')
    model_dir=data_dir/'models'/version
    model=Model.load(model_dir)
    catalog_source=catalog_source or data_dir/'population.sqlite'
    advanced_path=model_dir/'advanced.json'
    advanced=json.loads(advanced_path.read_text()) if advanced_path.exists() else None
    content_directory=None
    if advanced:
        from gamecompass.content import ContentModel
        content_directory=(model_dir/advanced['content_directory']).resolve()
        if not content_directory.is_relative_to(model_dir.resolve()):raise ValueError('Invalid structured content directory')
        ContentModel.load(content_directory)
    with tempfile.TemporaryDirectory() as tmp:
        catalog=Path(tmp)/'population.sqlite'
        with sqlite3.connect(f'file:{catalog_source.resolve()}?mode=ro',uri=True) as src, sqlite3.connect(catalog) as dst:
            schema=src.execute("SELECT sql FROM sqlite_master WHERE name='games'").fetchone()[0]
            dst.execute(schema)
            rows=src.execute('SELECT * FROM games')
            while batch:=rows.fetchmany(2000):
                dst.executemany('INSERT INTO games VALUES(?,?,?,?,?,?,?)',batch)
            dst.commit()
            game_count=dst.execute('SELECT count(*) FROM games').fetchone()[0]
        destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.write(catalog,'data/population.sqlite')
            z.writestr('data/models/current.json',json.dumps(pointer))
            for name in ('manifest.json','weights.npz','knn.npz','content.npz','evaluation.json','advanced.json'):
                if (model_dir/name).exists():z.write(model_dir/name,f'data/models/{version}/{name}')
            if content_directory:
                for name in ('manifest.json','weights.npz','tags.npz','words.npz'):
                    z.write(content_directory/name,f'data/models/{version}/{advanced["content_directory"]}/{name}')
            z.writestr('READY_README.txt',
                'GameCompass pretrained local runtime. Extract in the gamecompass source checkout.\n'
                'Install project dependencies and run: gamecompass app\n'
                'Contains cached Kaggle v28 publisher-declared CC0 metadata, Wikidata CC0 source entities when present, and learned item weights.\n'
                'Original game content retains applicable third-party rights; personal/noncommercial use.\n'
                'No local profiles, credentials, raw reviews, population user histories, or UCSD data.\n'
                'The included population.sqlite is catalog-only. To retrain or reproduce benchmarks,\n'
                'run gamecompass acquire --source kaggle --user-modulus 32 --activate first.\n'
                'Source/provenance/licenses and actual benchmark limitations are documented in the repo.\n')
            z.writestr('data/runtime-info.json',json.dumps({'model_version':version,'catalog_games':game_count,
                         'selected':model.selected,'contains_population_interactions':False,'contains_local_profiles':False},indent=2))
        with zipfile.ZipFile(destination) as z:
            if z.testzip() is not None:raise ValueError('Archive integrity check failed')
    return {'path':str(destination.resolve()),'bytes':destination.stat().st_size,'version':version,'catalog_games':game_count}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog-source',type=Path,help='Optional games-only source, e.g. verified enriched local.sqlite; profiles never copied')
    a=p.parse_args();print(json.dumps(package_runtime(a.data_dir,a.output,a.catalog_source),indent=2))
