import sqlite3
import pytest
from gamecompass.store import Store, GAME_SCHEMA
from gamecompass.learning import incorporate_feedback


def test_explicit_training_snapshot_is_opt_in_and_does_not_mutate_source(tmp_path):
    data=tmp_path/'source'; data.mkdir()
    with sqlite3.connect(data/'population.sqlite') as db:
        db.execute(GAME_SCHEMA)
        db.execute('CREATE TABLE interactions(user_id TEXT,game_id TEXT,positive INTEGER,timestamp INTEGER,playtime REAL,PRIMARY KEY(user_id,game_id))')
    store=Store(data);p=store.create_profile('Private')
    for i in range(3): store.upsert_game({'game_id':f'steam:{i}','title':str(i)})
    store.set_feedback(p,'steam:0',2,'played')
    store.set_feedback(p,'steam:1',None,'owned')
    store.set_feedback(p,'steam:2',-1,'played')
    out=tmp_path/'new-snapshot'
    result=incorporate_feedback(data,out)
    assert result['local_explicit_labels']==2
    with sqlite3.connect(data/'population.sqlite') as db: assert db.execute('SELECT count(*) FROM interactions').fetchone()[0]==0
    with sqlite3.connect(out/'population.sqlite') as db:
        assert sorted(r[0] for r in db.execute('SELECT positive FROM interactions'))==[0,1]
    with pytest.raises(ValueError): incorporate_feedback(data,out)
