from pathlib import Path

from streamlit.testing.v1 import AppTest

from gamecompass.store import Store

UI = Path(__file__).parents[1] / "src" / "gamecompass" / "ui.py"


def test_onboarding_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    app = AppTest.from_file(str(UI)).run()
    assert not app.exception
    app.text_input[0].set_value("Test Player")
    app.button[0].click().run()
    assert not app.exception
    assert Store(tmp_path).profiles()[0]["name"] == "Test Player"


def test_search_rate_and_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    store = Store(tmp_path)
    profile = store.create_profile("Test")
    store.upsert_game({"game_id": "steam:1", "title": "Wonderful Game"})
    app = AppTest.from_file(str(UI)).run()
    app.radio[0].set_value("Search").run()
    app.text_input[0].set_value("Wonderful").run()
    assert not app.exception
    assert app.subheader[0].value == "Wonderful Game"
    next(box for box in app.selectbox if box.label == "Your enjoyment").set_value("Loved it")
    next(box for box in app.selectbox if box.label == "Activity").set_value("completed")
    app.button[0].click().run()
    assert not app.exception
    assert store.feedback(profile)[0]["rating"] == 2
    assert store.feedback(profile)[0]["status"] == "completed"
    app.radio[0].set_value("Settings").run()
    assert not app.exception
    assert len(app.tabs) == 5


def test_discover_content_mode_and_separate_outcome(tmp_path, monkeypatch):
    from gamecompass.engine import Recommender
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    store = Store(tmp_path)
    profile = store.create_profile("Tester")
    store.upsert_game({"game_id": "steam:99", "title": "Unplayed Recommendation", "platforms": ["Windows"]})
    calls = []

    def recommend(self, profile_id, **kwargs):
        calls.append(kwargs)
        return [dict(store.game("steam:99"), explanation="Real-model output mocked for an isolated UI test.")]

    monkeypatch.setattr(Recommender, "recommend", recommend)
    app = AppTest.from_file(str(UI)).run()
    app.radio[0].set_value("Discover").run()
    next(box for box in app.checkbox if box.label.startswith("Explore by game descriptions")).set_value(True).run()
    assert calls[-1]["content_explore"] is True
    assert not app.exception
    next(control for control in app.select_slider if control.label == "Your feedback").set_value("Good")
    next(button for button in app.button if button.label == "Record evaluation").click().run()
    assert not app.exception
    outcomes = store.personal_history(profile)["outcomes"]
    assert outcomes[0]["kind"] == "recommendation_relevance"
    assert outcomes[0]["value"] == 1
    assert not store.feedback(profile)
