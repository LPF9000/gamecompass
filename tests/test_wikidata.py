import pytest

from gamecompass.wikidata import WikidataClient, WikidataError, statement_query, build_catalog, _date_iso


def row(prop, value, *, qid="Q1", app="10", label="Example Remaster", value_label=None, rank="NormalRank", precision=None, qualifiers=None):
    result = {"steam": {"value": app}, "item": {"value": "http://www.wikidata.org/entity/"+qid},
              "itemLabel": {"value": label}, "property": {"value": "http://www.wikidata.org/entity/"+prop},
              "statement": {"value": f"http://www.wikidata.org/entity/statement/{qid}-{prop}-{value}"},
              "value": {"value": value}, "rank": {"value": "http://wikiba.se/ontology#"+rank}}
    if value_label:
        result["valueLabel"] = {"value": value_label}
    if precision is not None:
        result["precision"] = {"value": str(precision)}
    if qualifiers:
        prop, val = qualifiers
        result["qualifierProperty"] = {"value": "http://www.wikidata.org/prop/qualifier/"+prop}
        result["qualifierValue"] = {"value": val}
    return result


def batch(rows):
    return {"query_sha256": "abc", "retrieved_at": "2026-11-11T00:00:00Z", "response": {"results": {"bindings": rows}}}


def test_no_work_to_steam_edition_inheritance():
    rows = [row("P1733", "10"), row("P31", "http://www.wikidata.org/entity/Q7889"),
            row("P629", "http://www.wikidata.org/entity/Q2"),
            row("P400", "http://www.wikidata.org/entity/Q5014725", value_label="PlayStation 4"),
            row("P577", "2016-05-12T00:00:00Z", precision=11, qualifiers=("P400","http://www.wikidata.org/entity/Q1406"))]
    steam = {"steam:10": {"title": "Example Remaster", "description": "Licensed description", "platforms": ["Windows"]}}
    games, reviews = build_catalog([batch(rows)], steam)
    assert len(games) == 1
    game = games[0]
    assert game["game_id"] == "wikidata:Q1"
    assert game["platforms"] == ["PlayStation 4"]
    assert steam["steam:10"]["platforms"] == ["Windows"]
    assert game["metadata"]["wikidata"]["version_of"] == ["Q2"]
    assert game["metadata"]["wikidata"]["original_work_release_date"] is None
    assert game["metadata"]["entity_release_date"] == "2016-05-12"
    assert "P400" in game["metadata"]["wikidata"]["publication_dates"][0]["qualifiers"]
    assert game["metadata"]["description_provenance"]["provider"] == "Kaggle Steam v28"


def test_ambiguous_mapping_and_deprecated_facts_are_not_silently_merged():
    rows=[]
    for qid in ("Q1","Q2"):
        rows += [row("P1733","10",qid=qid),row("P31","http://www.wikidata.org/entity/Q7889",qid=qid),
                 row("P400","http://www.wikidata.org/entity/Q100",qid=qid,value_label="Wrong Platform",rank="DeprecatedRank")]
    games,reviews=build_catalog([batch(rows)],{"steam:10":{"description":"Do not copy ambiguous description"}})
    assert len(games)==2
    assert reviews[0]["status"]=="review_multiple_entities"
    assert all(g["platforms"]==[] for g in games)
    assert all(g["description"]=="" for g in games)
    assert all(g["metadata"]["wikidata"]["original_work_release_date"] is None for g in games)


def test_date_precision_and_invalid_queries():
    assert _date_iso("1998-00-00T00:00:00Z",9)=="1998"
    assert _date_iso("1998-02-00T00:00:00Z",10)=="1998-02"
    assert _date_iso("1998-02-30T00:00:00Z",11) is None
    assert _date_iso("1998-01-01T00:00:00Z",8) is None
    with pytest.raises(ValueError):statement_query(["steam:10\" } UNION {"])
    with pytest.raises(ValueError):statement_query(["steam:10"]*26)
    assert '"10"' in statement_query(["steam:10"])


def test_api_denial_stops_without_querying_alternate_routes():
    class Response:
        status_code=403
        headers={}
    class Session:
        def __init__(self):self.calls=[]
        def get(self,url,**kwargs):self.calls.append((url,kwargs));return Response()
    session=Session()
    with pytest.raises(WikidataError,match="denied"):
        WikidataClient(session=session,sleep=lambda _:None).fetch(["steam:10"])
    assert len(session.calls)==1
    assert 'github.com/LPF9000/gamecompass' in session.calls[0][1]['headers']['User-Agent']


def test_api_retries_429_with_bounded_backoff():
    class Response:
        headers={"Retry-After":"2"}
        def __init__(self,status):self.status_code=status
        def json(self):return {"results":{"bindings":[]}}
    class Session:
        def __init__(self):self.calls=0
        def get(self,*args,**kwargs):self.calls+=1;return Response(429 if self.calls==1 else 200)
    sleeps=[];session=Session()
    result=WikidataClient(session=session,sleep=sleeps.append).fetch(["steam:10"])
    assert session.calls==2
    assert 2 in sleeps
    assert result['license']=='CC0-1.0'


def test_historical_steam_rename_does_not_copy_obsolete_description():
    rows = [row("P1733", "730", label="Counter-Strike 2", app="730"),
            row("P31", "http://www.wikidata.org/entity/Q7889", label="Counter-Strike 2", app="730")]
    games,_ = build_catalog([batch(rows)], {"steam:730":{"title":"Counter-Strike: Global Offensive","description":"Old release text","metadata":{"tags":["OldTag"]}}})
    assert games[0]["description"]==""
    assert games[0]["metadata"]["tags"]==[]


def test_mul_fallback_uses_official_labels_not_guessed_names(tmp_path):
    from gamecompass.wikidata import complete_labels
    rows=[row("P1733","10"), row("P31","http://www.wikidata.org/entity/Q7889",value_label="video game"),
          row("P400","http://www.wikidata.org/entity/Q63184502",value_label="Q63184502")]
    class Response:
        status_code=200
        headers={}
        def json(self):return {"results":{"bindings":[{"item":{"value":"http://www.wikidata.org/entity/Q63184502"},"label":{"value":"PlayStation 5"}}]}}
    class Session:
        def get(self,url,**kwargs):
            assert '"mul"' in kwargs['params']['query']
            return Response()
    client=WikidataClient(session=Session(),sleep=lambda _:None)
    labels=complete_labels([batch(rows)],tmp_path,client)
    games,_=build_catalog([batch(rows)],label_map=labels['labels'])
    assert games[0]['platforms']==['PlayStation 5']
    assert labels['query_sha256']


def test_apply_updates_games_without_changing_user_feedback(tmp_path):
    import hashlib
    from gamecompass.wikidata import acquire_enrichment
    from gamecompass.store import Store
    store=Store(tmp_path)
    store.upsert_game({'game_id':'steam:10','title':'Existing','platforms':['Windows']})
    profile=store.create_profile('Private tester')
    store.set_feedback(profile,'steam:10',rating=2,status='completed')
    before=store.export_profile(profile)
    class Client(WikidataClient):
        def fetch(self,ids):
            q=statement_query(ids)
            result=batch([row('P1733','10'),row('P31','http://www.wikidata.org/entity/Q7889',value_label='video game'),row('P400','http://www.wikidata.org/entity/Q63184502',value_label='PlayStation 5')])
            result.update(query_sha256=hashlib.sha256(q.encode()).hexdigest(),requested_game_ids=list(ids),query=q)
            return result
    report=acquire_enrichment(tmp_path,['steam:10'],client=Client(sleep=lambda _:None),apply=True)
    assert report['catalog_entities']==1
    assert store.game('wikidata:Q1')['platforms']==['PlayStation 5']
    assert store.game('steam:10')['platforms']==['Windows']
    assert store.export_profile(profile)==before


def test_relative_parent_work_date_requires_positive_relations_and_rejects_ambiguity():
    from gamecompass.wikidata import resolve_parent_work_date
    child={'game_id':'wikidata:Q1','metadata':{'wikidata':{'entity_id':'Q1','version_of':['Q2']}}}
    relation=[{'property':'P629','value':'http://www.wikidata.org/entity/Q2','statement':'child-P629-parent','rank':'NormalRank'}]
    facts=[{'property':'P31','value':'http://www.wikidata.org/entity/Q7889','statement':'parent-type','rank':'NormalRank'},
           {'property':'P747','value':'http://www.wikidata.org/entity/Q1','statement':'parent-P747-child','rank':'NormalRank'},
           {'property':'P577','value':'1999-09-27T00:00:00Z','precision':11,'statement':'parent-publication','rank':'NormalRank','qualifiers':{'P400':['http://www.wikidata.org/entity/Q1406']},'reference_urls':[]}]
    decision=resolve_parent_work_date(child,facts,relation)
    assert decision['date']['iso']=='1999-09-27'
    assert decision['date']['parent_entity_id']=='Q2'
    assert decision['date']['relation_statement']=='child-P629-parent'
    assert decision['date']['publication_statement']=='parent-publication'
    assert 'not proof of ultimate original' in decision['date']['semantics']
    assert resolve_parent_work_date(child,facts,[])['date'] is None
    assert resolve_parent_work_date(child,facts[:2],relation)['date'] is None
    edition=facts+[{'property':'P629','value':'http://www.wikidata.org/entity/Q3','statement':'parent-is-edition','rank':'NormalRank'}]
    assert resolve_parent_work_date(child,edition,relation)['date'] is None
    contradiction=facts+[{**facts[-1],'value':'1998-09-27T00:00:00Z','statement':'contradictory-date'}]
    assert resolve_parent_work_date(child,contradiction,relation)['date'] is None
    series=[{**facts[0],'value':'http://www.wikidata.org/entity/Q7058673'},*facts[1:]]
    assert resolve_parent_work_date(child,series,relation)['date'] is None
