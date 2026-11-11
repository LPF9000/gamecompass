import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('train_recbole', Path(__file__).parents[1] / 'scripts/train_recbole.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_atomic_export_preserves_semantics(tmp_path):
    path = tmp_path / 'test.inter'
    result = module.export_atomic([('u','steam:1',1,100),('u','steam:2',0,200)],path)
    assert result['positive_interactions']==1
    assert path.read_text().splitlines()==['user_id:token\titem_id:token\ttimestamp:float','u\tsteam:1\t100']
    assert len(result['sha256'])==64


def test_export_rejects_invalid_identifiers(tmp_path):
    with pytest.raises(ValueError): module.export_atomic([('u\n','steam:1',1,None)],tmp_path/'x.inter')


def test_resume_rejects_wrong_source_and_corrupt_checkpoint(tmp_path):
    path = tmp_path / 'checkpoints' / 'own.pth'
    path.parent.mkdir()
    path.write_bytes(b'local-test-fixture')
    signature = {'algorithm':'BPR','atomic_export_sha256':'source'}
    module.write_json(path.with_suffix('.pth.json'), {'signature':signature,'checkpoint_sha256':module.file_sha256(path),'epoch':0})
    assert module.validate_resume(path,path.parent,signature)['epoch']==0
    with pytest.raises(ValueError,match='source export'):
        module.validate_resume(path,path.parent,{'algorithm':'BPR','atomic_export_sha256':'different'})
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):
        module.validate_resume(path,path.parent,signature)
    with pytest.raises(ValueError,match='local checkpoint'):
        module.validate_resume(path,tmp_path/'other',signature)


def test_optional_real_recbole_resume_repeats_uninterrupted_training(tmp_path):
    """Synthetic fixture verifies training mechanics only, never benchmark quality."""
    pytest.importorskip('torch')
    pytest.importorskip('recbole')
    import numpy as np
    atomic=tmp_path/'probe'
    module.export_atomic([('u1','steam:1',1,1),('u1','steam:2',1,2),('u2','steam:2',1,1),('u2','steam:3',1,2),('u3','steam:3',1,1),('u3','steam:4',1,2)],atomic/'probe.inter')
    _,_,full=module.train_one('BPR',atomic,tmp_path/'full',2,42)
    module.train_one('BPR',atomic,tmp_path/'partial',1,42)
    checkpoint=next((tmp_path/'partial/checkpoints').glob('*.pth'))
    _,_,resumed=module.train_one('BPR',atomic,tmp_path/'partial',2,42,resume_checkpoint=checkpoint)
    assert np.array_equal(full,resumed)
