"""Create an immutable v2 runtime from frozen, validation-selected improvements."""
from pathlib import Path
import json,shutil,argparse
from gamecompass.models import Model,atomic_json,sha256
from gamecompass.content import ContentModel


def activate(data_dir:Path,content_path:Path,version:str='validated-v2-20261110')->Path:
    base_version=json.loads((data_dir/'protocols/fresh-users-v2.json').read_text())['base_version']
    base=data_dir/'models'/base_version
    Model.load(base);ContentModel.load(content_path)
    frozen=json.loads((data_dir/'protocols/v2-confirmation.frozen-config.json').read_text())
    if sha256(content_path/'manifest.json')!=frozen['config']['content_manifest_sha256']:raise ValueError('Content differs from frozen config')
    report=json.loads((data_dir/'fresh-v2-results.json').read_text())
    if report['frozen_config']!=frozen:raise ValueError('Confirmation provenance differs')
    if Path(version).name!=version:raise ValueError('Version must be a directory name')
    destination=data_dir/'models'/version
    if destination.exists():raise ValueError('Use a fresh immutable version name')
    shutil.copytree(base,destination)
    shutil.copytree(content_path,destination/'structured-content')
    manifest=json.loads((destination/'manifest.json').read_text())
    manifest.update(version=version,base_version=base_version,
                    followup='Validation-selected structured content and bounded mixed profile; main CF unchanged')
    atomic_json(destination/'manifest.json',manifest)
    mixed=frozen['config']['mixed']
    atomic_json(destination/'advanced.json',{'format_version':1,'content_directory':'structured-content','mixed':mixed,
        'mixed_scope':{'minimum_total_cf':5,'maximum_total_cf':20,'minimum_explicit':1,'maximum_explicit':3},
        'content_manifest_sha256':sha256(destination/'structured-content/manifest.json'),
        'global_blend_promoted':False,'selection_source':'validation-only choices frozen before fresh confirmation',
        'confirmation_protocol_sha256':frozen['protocol_sha256']})
    Model.load(destination);ContentModel.load(destination/'structured-content')
    atomic_json(data_dir/'models/current.json',{'version':version})
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--content-path',type=Path,required=True)
    a=p.parse_args();print(activate(a.data_dir,a.content_path))
