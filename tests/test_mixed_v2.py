"""Numerical and leakage fixtures; these are not real benchmark results."""
import importlib.util
from pathlib import Path
import sqlite3
import pytest
from gamecompass.evaluation import user_partition
from gamecompass.mixed import MixedPersonalizer

spec=importlib.util.spec_from_file_location('mixed_benchmark',Path(__file__).parents[1]/'scripts'/'benchmark_mixed_v2.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_explicit_negative_neutral_and_not_interested_override_weak_evidence():
    support=MixedPersonalizer.combine({'bad':-1,'neutral':0,'good':2,'conflict':1},
                                      {'bad':100,'neutral':50,'good':200,'other':10,'conflict':1000},
                                      .25,'uniform',not_interested=['conflict'])
    assert support=={'bad':-1,'neutral':0,'good':2,'other':.25}
    assert MixedPersonalizer.combine({'a':1},{'b':100},0,'uniform')=={'a':1}


def test_log_hours_normalized_and_zero_playtime_never_becomes_like():
    support=MixedPersonalizer.combine({}, {'long':99,'short':9,'zero':0},.5,'normalized_log1p_hours')
    assert support['long']==.5
    assert support['short']==pytest.approx(.25)
    assert 'zero' not in support
    assert MixedPersonalizer.combine({}, {'zero':0},1,'normalized_log1p_hours')=={}


@pytest.mark.parametrize('coefficient,weighting',[(float('nan'),'uniform'),(-.1,'uniform'),(1.1,'uniform'),(.2,'unknown')])
def test_invalid_settings_fail(coefficient,weighting):
    with pytest.raises(ValueError): MixedPersonalizer.combine({}, {},coefficient,weighting)


def test_invalid_hours_and_explicit_values_fail():
    with pytest.raises(ValueError): MixedPersonalizer.combine({}, {'bad':-1},.5,'uniform')
    with pytest.raises(ValueError): MixedPersonalizer.combine({'bad':float('inf')},{},.5,'uniform')


def test_validation_queries_never_retrieve_other_partition_labels(tmp_path):
    users={split:next(f'user{i}' for i in range(10000) if user_partition(f'user{i}')==split)
           for split in ('train','validation','test')}
    source=tmp_path/'population.sqlite'
    ids=[f'game{i:02}' for i in range(25)]
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE interactions(user_id TEXT,game_id TEXT,positive INTEGER,timestamp INTEGER,playtime REAL)')
        for split,user in users.items():
            db.executemany('INSERT INTO interactions VALUES(?,?,?,?,?)',
                           [(user,g,int(i%4!=0),i,float(i)) for i,g in enumerate(ids)])
    cohorts,stats=benchmark.validation_queries(source,ids,max_users=100)
    assert stats['source_users']==3 and stats['original_validation_users']==1
    for group,queries in cohorts.items():
        n,k=map(int,group.split(':'))
        assert len(queries)==1
        query=queries[0]
        assert query['user_id']==users['validation']
        assert len(query['explicit'])==k and len(query['implicit_hours'])==n
        assert len(query['seen'])==n and query['target'].isdisjoint(query['seen'])
        assert set(query['explicit']).issubset(query['implicit_hours'])
        # Hours-only records carry no hidden recommendation label.
        assert query['implicit_hours'][ids[0]]==0.
