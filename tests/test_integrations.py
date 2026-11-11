from unittest.mock import Mock
import pytest
from gamecompass.integrations import SteamProvider, ProviderError


def client(payload, status=200):
    session = Mock()
    session.get.return_value.status_code = status
    session.get.return_value.json.return_value = payload
    return SteamProvider('secret-test-key', session), session


def test_owned_playtime_is_not_rating():
    p,s = client({'response':{'game_count':1,'games':[{'appid':10,'name':'Game','playtime_forever':30}]}})
    g=p.library('76561198000000000')[0]
    assert g['game_id']=='steam:10' and g['playtime_minutes']==30
    assert 'rating' not in g
    assert s.get.call_args.kwargs['timeout']==25


def test_private_and_missing_credentials(monkeypatch):
    monkeypatch.delenv('STEAM_API_KEY', raising=False)
    with pytest.raises(ProviderError): SteamProvider()
    p,_=client({'response':{}})
    with pytest.raises(ProviderError,match='visibility'): p.library('76561198000000000')


def test_validate_id_and_secret_redaction():
    p,_=client({},403)
    with pytest.raises(ValueError): p.library('user')
    with pytest.raises(ProviderError) as e: p.library('76561198000000000')
    assert 'secret-test-key' not in str(e.value)


def test_recent_and_achievements():
    p,_=client({'playerstats':{'success':True,'achievements':[{'achieved':1}]}})
    assert p.achievements('76561198000000000',10)==[{'achieved':1}]
