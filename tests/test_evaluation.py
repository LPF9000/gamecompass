import sqlite3
import pytest
from gamecompass.evaluation import user_partition, prepare, cold_queries, ranking_metrics, interaction_matrix


def test_ranking_metric_known_values():
    result = ranking_metrics(["a","b","c"], {"a","c"}, 2)
    assert result["recall@2"] == .5
    assert result["map@2"] == .5
    assert 0 < result["ndcg@2"] < 1
    with pytest.raises(ValueError): ranking_metrics([], set(), 10)


def test_support_targets_and_negatives_separate():
    rows = [("u","a",1,1),("u","b",0,2),("u","c",1,3)]
    support, seen, targets, negatives = cold_queries({"u":rows},2)[0]
    assert support == {"a":1,"b":-1}
    assert seen == {"a","b"}
    assert targets == {"c"}
    assert not targets & seen
    assert cold_queries({"u":rows},3) == []


def test_no_test_interaction_or_item_scope_leakage(tmp_path):
    train = next(f"u{i}" for i in range(1000) if user_partition(f"u{i}") == "train")
    test = next(f"u{i}" for i in range(1000) if user_partition(f"u{i}") == "test")
    val = next(f"u{i}" for i in range(1000) if user_partition(f"u{i}") == "validation")
    db=sqlite3.connect(tmp_path/'population.sqlite')
    db.executescript('CREATE TABLE games(game_id TEXT PRIMARY KEY,title TEXT,description TEXT); CREATE TABLE interactions(user_id TEXT,game_id TEXT,positive INTEGER,timestamp INTEGER);')
    db.executemany('INSERT INTO games VALUES(?,?,?)',[(g,g,'') for g in ['a','b','testonly']])
    db.executemany('INSERT INTO interactions VALUES(?,?,?,?)',[(train,'a',1,1),(train,'b',1,2),(test,'testonly',1,1),(val,'a',1,1)])
    db.commit();db.close()
    prepared=prepare(tmp_path,max_items=10)
    assert 'testonly' not in prepared['cf_ids']
    assert all(r[0] != test for r in prepared['train_rows'])
    matrix=interaction_matrix(prepared['train_rows'],prepared['cf_ids'])
    assert matrix.sum()==2


def test_implicit_support_hides_recommendation_labels():
    from gamecompass.evaluation import implicit_queries
    rows=[('u','a',0,1,100),('u','b',1,2,1),('u','c',1,3,5)]
    uniform=implicit_queries({'u':rows},2,'uniform')[0]
    weighted=implicit_queries({'u':rows},2,'log1p_hours')[0]
    explicit=implicit_queries({'u':rows},2,'explicit')[0]
    assert uniform[0] == {'a':1,'b':1}
    assert weighted[0]['a'] > weighted[0]['b'] > 0
    assert explicit[0]['a'] == -1
    assert uniform[2] == {'c'}
    assert not uniform[1] & uniform[2]


def test_cold_items_never_enter_support_and_targets_follow_prefix():
    from gamecompass.evaluation import cold_item_queries, cold_item_partition
    rows=[('u','cold_before',1,1),('u','warm_a',1,2),('u','warm_b',0,3),('u','cold_after',1,4)]
    cold={'cold_before','cold_after'}
    query=cold_item_queries({'u':rows},cold,2)[0]
    assert query[0] == {'warm_a':1,'warm_b':-1}
    assert query[2] == {'cold_after'}
    assert not set(query[0]) & cold
    assert cold_item_partition('steam:10',42) == cold_item_partition('steam:10',42)
    with pytest.raises(ValueError):cold_item_queries({'u':rows},cold,0)


def test_analytic_random_expectation_matches_all_small_rankings():
    from itertools import permutations
    from gamecompass.evaluation import expected_random_metrics
    target={'a','b'}
    metrics=[ranking_metrics(list(p),target,3) for p in permutations(['a','b','c','d'])]
    expected=expected_random_metrics(2,4,3)
    for key,value in expected.items():
        assert sum(m[key] for m in metrics)/len(metrics) == pytest.approx(value)
