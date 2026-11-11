"""Loopback-only Streamlit interface; all persistence is delegated to Store."""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from gamecompass.store import RATING_LABELS, REASONS, STATUSES, Store
from gamecompass.refinements import RecommendationFilters, catalog_filter_options, genre_facts, release_facts

RATINGS = ["Haven’t played enough to judge", "Loved it", "Liked it", "Neutral", "Disliked it", "Hated it"]
LABEL_TO_RATING = {label: value for value, label in RATING_LABELS.items()}


def rating_editor(store: Store, profile_id: str, game: dict[str, Any], prefix: str) -> None:
    game_id = game["game_id"]
    existing = next((row for row in store.feedback(profile_id) if row["game_id"] == game_id), {})
    with st.expander("Rate or update activity", expanded=False):
        with st.form(f"{prefix}:{profile_id}:{game_id}"):
            rating_label = RATING_LABELS.get(existing.get("rating"), RATINGS[0])
            label = st.selectbox("Your enjoyment", RATINGS, index=RATINGS.index(rating_label))
            state = st.selectbox("Activity", STATUSES, index=STATUSES.index(existing.get("status", "unplayed")))
            exclude = st.checkbox("Not interested — exclude from recommendations", value=existing.get("not_interested", False))
            reasons = st.multiselect("What shaped your rating? (optional)", REASONS, default=existing.get("reasons", []))
            if st.form_submit_button("Save feedback", use_container_width=True):
                store.set_feedback(profile_id, game_id, LABEL_TO_RATING.get(label), state, exclude, reasons)
                st.success("Saved. Your next recommendations use this feedback.")
                st.rerun()


def game_card(store: Store, profile_id: str, game: dict[str, Any], prefix: str, rank: int | None = None) -> None:
    with st.container(border=True):
        st.subheader(f"{rank}. {game['title']}" if rank else game["title"])
        tags = game.get("platforms", []) + game.get("genres", [])
        if tags:
            st.caption(" · ".join(tags[:10]))
        description = game.get("description", "")
        if description:
            st.write(description[:900])
        if game.get("explanation"):
            st.info(game["explanation"])
        catalog_tags = game.get('metadata',{}).get('tags',[])
        if isinstance(catalog_tags,list) and catalog_tags:
            st.caption('Catalog tags: ' + ', '.join(str(t) for t in catalog_tags[:8]))
        if game.get("themes"):
            st.caption("Themes: " + ", ".join(game["themes"][:8]))
        if game.get("uncertainty"):
            st.caption(game["uncertainty"])
        inferred_genres = [fact["genre"] for fact in genre_facts(game) if fact["source"] != "catalog"]
        if inferred_genres:
            st.caption("Genres inferred from catalog tags: " + ", ".join(inferred_genres[:6]))
        facts = game.get('release_context') or release_facts(game)
        if facts["original_year"] or facts["edition_year"] or facts["release_kind"] != "unknown":
            st.caption(f"Original work: {facts['original_year'] or 'unknown'} · This edition/listing: {facts['edition_year'] or 'unknown'} · Release type: {facts['release_kind']}")
        if facts.get('platform'):
            st.caption(f"Release-date context: {facts['platform']} · {facts.get('edition_source','')}")
        rating_editor(store, profile_id, game, prefix)
        if prefix != "similar" and st.button("If you liked this game", key=f"similar-seed:{prefix}:{game['game_id']}"):
            st.session_state["similar_seed"] = game["game_id"]
            st.rerun()
        if prefix == "discover":
            with st.expander("Evaluate this recommendation"):
                with st.form(f"outcome:{profile_id}:{game['game_id']}"):
                    kind = st.selectbox("What are you evaluating?", ["How relevant it looks", "Enjoyment after playing"])
                    options = ["Very poor", "Poor", "Neutral", "Good", "Excellent"]
                    value = st.select_slider("Your feedback", options=options, value="Neutral")
                    st.caption("Relevance and enjoyment after playing are tracked separately from your game rating.")
                    if st.form_submit_button("Record evaluation"):
                        from gamecompass.personal import record_outcome
                        record_outcome(store.data_dir, profile_id, game["game_id"],
                                       "recommendation_relevance" if kind == "How relevant it looks" else "post_play_enjoyment",
                                       options.index(value) - 2)
                        st.success("Recorded locally.")


def recommendations(store: Store, profile_id: str, limit: int = 12,
                    platform: str | None = None, exclude_played: bool = True,
                    content_explore: bool = False,
                    filters: RecommendationFilters | None = None) -> list[dict[str, Any]]:
    try:
        from gamecompass.engine import Recommender
        with st.spinner("Updating your recommendations…"):
            options = {"content_explore": True} if content_explore else {}
            if filters is not None:
                options["filters"] = filters
            return Recommender(store.data_dir).recommend(profile_id, limit=limit, platform=platform, exclude_played=exclude_played, **options)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        st.warning(str(error))
        st.caption("Prepare real data and train a population model using the commands in the README. You can already search and rate any games in your local catalog.")
        return []
    except ImportError:
        st.warning("Recommendation engine is not available in this installation. Check the setup guide.")
        return []



@st.cache_data(show_spinner=False)
def _filter_options(data_dir: str, database_signature: tuple[int, int]) -> dict[str, list[str]]:
    return catalog_filter_options(Store(Path(data_dir)))


def discover_filters(store: Store) -> RecommendationFilters:
    signature = store.path.stat()
    options = _filter_options(str(store.data_dir.resolve()), (signature.st_mtime_ns, signature.st_size))
    columns = st.columns(3)
    with columns[0]:
        platform = st.selectbox("Available platform", ["All platforms", *options["platforms"]], key="discover_platform")
    with columns[1]:
        genre = st.selectbox("Genre", ["Any genre", *options["genres"]], key="discover_genre")
    eras = {"Any era": (None, None), "Before 1990": (1950, 1989), "1990s": (1990, 1999),
            "2000s": (2000, 2009), "2010s": (2010, 2019), "2020s": (2020, 2029)}
    with columns[2]:
        era = st.selectbox("Era", list(eras), key="discover_era")
    era_basis = "original"
    with st.expander("Edition details"):
        if era != "Any era":
            use_original = st.checkbox("Use the original work’s date instead of this release’s date", value=False, key="discover_original_date")
            era_basis = "original" if use_original else "edition"
        policies = {"Any release type": "any", "Original versions only": "originals_only", "Remakes and remasters only": "remakes_remasters_only"}
        edition_policy = st.selectbox("Release type", list(policies), key="discover_edition_policy")
        st.caption("Original-release dates can differ from later editions or ports. A specific release-type filter omits records whose type is unknown.")
    st.caption("Ratings inform your taste across the whole catalog. These controls narrow recommendations only.")
    if not any("PlayStation" in value or "Xbox" in value or "Nintendo" in value for value in options["platforms"]):
        st.caption("This catalog currently has no verified console availability. Console choices appear after verified metadata is added; missing availability is not treated as incompatibility.")
    if era != "Any era":
        st.caption("Games with an unknown " + ("original-work" if era_basis == "original" else "edition/listing") + " date are excluded by this era filter. Era normally uses this release’s date. Choose original-work dates in Edition details when you want that stricter view.")
    lower, upper = eras[era]
    return RecommendationFilters(platform=None if platform == "All platforms" else platform,
                                 genre=None if genre == "Any genre" else genre,
                                 year_min=lower, year_max=upper, era_basis=era_basis,
                                 edition_policy=policies[edition_policy])


def clear_similar() -> None:
    st.session_state.pop("similar_seed", None)


def similar_panel(store: Store, profile_id: str, game_id: str) -> None:
    game = store.game(game_id)
    if not game:
        st.warning("This source game is no longer in the local catalog.")
        st.button("Back to browsing", on_click=clear_similar)
        return
    st.title("If you liked " + game["title"])
    st.caption("The selected game is the recommendation seed. This does not change your saved ratings. Your profile’s exclusions still apply.")
    st.button("Back to browsing", on_click=clear_similar)
    filters = discover_filters(store)
    try:
        from gamecompass.engine import Recommender
        with st.spinner("Finding games from this starting point…"):
            games = Recommender(store.data_dir).similar(game_id, limit=20, filters=filters, profile_id=profile_id)
        if not games:
            st.info("No games satisfy these refinements and exclusions. Broaden the filters or add verified metadata.")
        for index, candidate in enumerate(games, 1):
            game_card(store, profile_id, candidate, "similar", rank=index)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        st.warning(str(error))


def import_csv(store: Store, profile_id: str, content: str) -> int:
    """Import explicit manual ratings; canonical IDs are required to avoid fuzzy merges."""
    rows = list(csv.DictReader(io.StringIO(content)))
    if len(rows) > 100000:
        raise ValueError("CSV exceeds 100,000 rows")
    games, feedback, implicit = [], [], []
    for row in rows:
        game_id = row.get("game_id", "")
        game = store.game(game_id)
        if not game:
            title = row.get("title", "").strip()
            if not title:
                raise ValueError(f"Unknown game {game_id!r}: supply a title and canonical ID")
            game = {"game_id": game_id, "title": title}
        games.append(game)
        if row.get('playtime_minutes', '').strip():
            implicit.append({'game_id':game_id,'provider':'manual','playtime_minutes':float(row['playtime_minutes']),'metadata':'{}'})
        raw = row.get("rating", "").strip()
        feedback.append({"game_id": game_id, "rating": int(raw) if raw else None,
                         "status": row.get("status") or "unplayed",
                         "not_interested": row.get("not_interested", "").lower() in ("true", "1", "yes"), "reasons": []})
    # Reuse full import validation, then transfer the validated preferences into
    # the selected profile. Temporary profile deletion also clears private rows.
    payload = {"schema_version": 1, "profile": {"name": "CSV validation"}, "games": games, "feedback": feedback, "implicit_signals":implicit}
    temporary = store.import_profile(payload)
    try:
        signals = store.implicit(temporary)
        if signals:
            store.import_library(profile_id,[{**store.game(row['game_id']),'playtime_minutes':row['playtime_minutes']} for row in signals],provider='manual')
        for row in store.feedback(temporary):
            store.set_feedback(profile_id, row["game_id"], row["rating"], row["status"], row["not_interested"], row["reasons"])
    finally:
        store.delete_profile(temporary)
    return len(rows)


def settings(store: Store, profile_id: str) -> None:
    st.title("Settings")
    profiles_tab, accounts_tab, data_tab, evaluation_tab, diagnostics_tab = st.tabs(["Profiles", "Integrations", "Import & export", "Personal evaluation", "Model & privacy"])
    with profiles_tab:
        with st.form("new-profile"):
            name = st.text_input("New profile name", max_chars=80)
            if st.form_submit_button("Create independent profile"):
                try:
                    store.create_profile(name)
                    st.success("Created. Choose the new profile in the sidebar.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        st.divider()
        st.write("Delete this profile and its ratings, library signals, and connection records. The game catalog and population model are shared and remain available.")
        confirmed = st.checkbox("I understand profile deletion is permanent")
        if st.button("Delete selected profile", disabled=not confirmed, type="secondary"):
            store.delete_profile(profile_id)
            st.session_state.pop("selected_profile", None)
            st.rerun()
    with accounts_tab:
        st.subheader("Steam")
        connections = store.profile_connections(profile_id)
        for connection in connections:
            date = datetime.fromtimestamp(connection["imported_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"{connection['provider']}: {connection['game_count']} games imported · {date}")
        if not connections:
            st.caption("No gaming history imported. Manual ratings provide full onboarding.")
        st.write("Set STEAM_API_KEY in the launch environment. Enter your 17-digit Steam ID. Private libraries may be unavailable through the official API.")
        with st.form("steam-import"):
            steam_id = st.text_input("Steam ID", max_chars=17, help="Your public SteamID64, not a password or vanity URL")
            if st.form_submit_button("Import Steam library"):
                try:
                    from gamecompass.integrations import SteamProvider
                    with st.spinner("Importing authorized Steam history…"):
                        games = SteamProvider().library(steam_id)
                        count = store.import_library(profile_id, games, provider="steam")
                    st.success(f"Imported {count} games. Playtime is stored as activity, separately from your ratings.")
                except (ValueError, OSError) as error:
                    st.error(str(error))
                except ImportError:
                    st.error("Steam integration is not installed.")
                except Exception as error:
                    # ProviderError messages are safe; never display raw requests
                    # exceptions, which can contain API keys in their URLs.
                    if error.__class__.__name__ == "ProviderError":
                        st.error(str(error))
                    else:
                        st.error("Steam import failed. Check credentials and public profile visibility.")
        with st.expander("Recently played games and achievements"):
            st.caption("These are optional live API checks. Your Steam ID is used for this request and is not saved. Achievement lookup makes one game request, with no library-wide fanout.")
            recent_id = st.text_input("Steam ID for activity checks", max_chars=17)
            if st.button("View recently played games"):
                try:
                    from gamecompass.integrations import SteamProvider
                    with st.spinner("Checking recently played games…"):
                        recent_games = SteamProvider().recent(recent_id)
                    if recent_games:
                        st.dataframe([{key: item.get(key) for key in ("appid", "name", "playtime_2weeks", "playtime_forever")} for item in recent_games], hide_index=True)
                        st.caption("Playtime is in minutes; activity does not establish enjoyment.")
                    else:
                        st.info("No recent activity was exposed by Steam.")
                except Exception as error:
                    st.error(str(error) if error.__class__.__name__ in ("ProviderError", "ValueError") else "Steam activity check failed.")
            app_id = st.number_input("Steam app ID for achievement lookup", min_value=1, step=1, value=10)
            if st.button("View achievements for this game"):
                try:
                    from gamecompass.integrations import SteamProvider
                    with st.spinner("Checking this game’s achievements…"):
                        achievements = SteamProvider().achievements(recent_id, int(app_id))
                    if achievements:
                        st.dataframe(achievements, hide_index=True)
                    else:
                        st.info("No achievements were exposed for this game.")
                except Exception as error:
                    st.error(str(error) if error.__class__.__name__ in ("ProviderError", "ValueError") else "Steam achievement check failed.")
    with data_tab:
        with st.expander('Add a missing game or new release'):
            st.caption('Use a verified canonical service ID. Manual descriptions enable immediate content exploration; this does not create collaborative evidence.')
            with st.form('manual-catalog-entry'):
                canonical_id = st.text_input('Canonical game ID',placeholder='steam:12345')
                title = st.text_input('Game title')
                description = st.text_area('Game description')
                platform_names = st.text_input('Verified platforms (comma-separated)')
                genre_names = st.text_input('Genres (comma-separated)')
                theme_names = st.text_input('Themes (comma-separated)')
                alternate_titles = st.text_input('Alternative titles (comma-separated)')
                original_date = st.text_input('Original work release date (optional)', placeholder='1997 or 1997-01-31')
                edition_date = st.text_input('This edition release date (optional)', placeholder='2013 or 2013-12-05')
                release_kind = st.selectbox('Known release type', ['unknown', 'original', 'remake', 'remaster', 'rerelease'])
                st.caption('Enter dates and release type only when verified. These assertions are labeled user-provided; original and edition dates are not interchangeable.')
                update_existing = st.checkbox('Update an existing catalog record with this exact ID')
                if st.form_submit_button('Save catalog game'):
                    try:
                        if store.game(canonical_id) and not update_existing:
                            raise ValueError('That canonical ID exists. Review it and explicitly select update to replace its metadata.')
                        for value in (original_date.strip(), edition_date.strip()):
                            if value and not (len(value) == 4 and value.isdigit() and 1950 <= int(value) <= 2200):
                                try:
                                    parsed_date = datetime.strptime(value, '%Y-%m-%d').date()
                                except ValueError:
                                    raise ValueError('Release dates must be a year or a valid YYYY-MM-DD date') from None
                                if not 1950 <= parsed_date.year <= 2200:
                                    raise ValueError('Release years must be between 1950 and 2200')
                        split = lambda text: [part.strip() for part in text.split(',') if part.strip()]
                        store.upsert_game({'game_id':canonical_id,'title':title,'description':description,
                            'platforms':split(platform_names),'genres':split(genre_names),'themes':split(theme_names),
                            'metadata':{'source':'manual','identity_confidence':'user-supplied','alternative_titles':split(alternate_titles),
                                        'manual_release':{'source':'user-provided','original_date':original_date.strip() or None,
                                                          'edition_date':edition_date.strip() or None,'release_kind':release_kind}}})
                        st.success('Saved locally. Search and content exploration can use this game immediately.')
                    except ValueError as error:
                        st.error(str(error))
        st.subheader("Export this profile")
        st.download_button("Download private profile JSON", data=json.dumps(store.export_profile(profile_id), indent=2),
                           file_name="gamecompass-profile.json", mime="application/json")
        st.caption("Contains your preferences and imported activity. Keep it private. Credentials are never included.")
        st.subheader("Import a profile or manual ratings")
        uploaded = st.file_uploader("Profile JSON or ratings CSV", type=["json", "csv"])
        st.caption("CSV columns: game_id,title,rating,status,not_interested. Ratings use -2 (hated) through 2 (loved); blank means insufficient experience. CSV updates this profile; JSON creates an independent profile.")
        if uploaded and st.button("Import selected file"):
            if uploaded.size > 10 * 1024 * 1024:
                st.error("Import must be at most 10 MiB.")
            else:
                try:
                    content = uploaded.getvalue().decode("utf-8-sig")
                    if uploaded.name.lower().endswith(".json"):
                        store.import_profile(json.loads(content))
                        st.success("Imported an independent profile. Select it in the sidebar.")
                    else:
                        count = import_csv(store, profile_id, content)
                        st.success(f"Imported {count} feedback records.")
                except (ValueError, TypeError, KeyError, UnicodeError) as error:
                    st.error(f"Import rejected: {error}")
    with evaluation_tab:
        st.subheader("Check hidden preferences")
        st.write("Temporarily hide some of your known ratings from personalization, then check where the model ranks them. This never changes your saved ratings.")
        st.caption("Requires at least five mapped nonneutral ratings. This small diagnostic predicts preferences for already-played games; it does not measure enjoyment of unplayed recommendations. Use Discover’s evaluation controls to record relevance now and enjoyment after playing.")
        seed = st.number_input("Reproducible split seed", min_value=0, max_value=2147483647, value=42, step=1)
        if st.button("Run local preference holdout"):
            try:
                from gamecompass.personal import evaluate_profile
                with st.spinner("Evaluating withheld preferences…"):
                    report = evaluate_profile(store.data_dir, profile_id, seed=int(seed))
                st.json(report)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(str(error))
        for heading, rows in store.personal_history(profile_id).items():
            if rows:
                with st.expander(heading.replace("_", " ").title()):
                    st.json(rows[-20:])
    with diagnostics_tab:
        st.subheader("Local and private")
        st.write("Profiles and activity stay on this computer. No data is uploaded for training. Steam connects only when you request an import. Recommendations use downloaded artifacts and metadata offline.")
        st.caption("Anyone with access to your computer and data directory can read the local database. Profile deletion does not remove exports or backups you saved elsewhere.")
        st.metric("Games in local catalog", f"{store.catalog_count():,}")
        st.code(str(store.data_dir.resolve()), language=None)
        pointer = store.data_dir / "models" / "current.json"
        if pointer.exists():
            try:
                selected = json.loads(pointer.read_text())
                st.json(selected)
                version = selected.get('version','')
                if Path(version).name == version:
                    manifest = store.data_dir / 'models' / version / 'manifest.json'
                    if manifest.exists():
                        info = json.loads(manifest.read_text())
                        st.json({key:info.get(key) for key in ('selected','selection_by_history','training_users','training_positives','source_sha256','scope','implicit')})
            except (OSError, ValueError):
                st.warning("The current model pointer cannot be read. Restore a verified checkpoint.")
        else:
            st.caption("No production model selected yet.")
        files = ['active-population.json','runtime-info.json']
        active = store.data_dir / 'active-population.json'
        if active.exists():
            try:
                source = json.loads(active.read_text()).get('source',{})
                files.append('kaggle-population.stats.json' if source.get('ref') == 'antonkozyriev/game-recommendations-on-steam' else 'dataset-stats.json')
            except (OSError,ValueError):
                pass
        for filename in files:
            path = store.data_dir / filename
            if path.exists() and path.stat().st_size < 100000:
                with st.expander(filename):
                    try:
                        st.json(json.loads(path.read_text()))
                    except (OSError, ValueError):
                        st.warning("This diagnostic file is not valid JSON.")


def main() -> None:
    st.set_page_config(page_title="GameCompass", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .stApp {background:linear-gradient(155deg,#f6f8fc 0%,#f0f5fa 70%,#e4f3ef 100%);}
    [data-testid="stSidebar"] {background:#edf2f8;}
    [data-testid="stMetric"] {background:#e6eef7;border-radius:12px;padding:12px;}
    h1,h2,h3 {letter-spacing:-.025em;}
    div[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:14px;}
    </style>""", unsafe_allow_html=True)
    data_dir = Path(os.environ.get("GAMECOMPASS_DATA", "data"))
    store = Store(data_dir)
    with st.sidebar:
        st.title("◈ GameCompass")
        st.caption("Your next favorite, discovered locally.")
        profiles = store.profiles()
        if not profiles:
            st.subheader("Start with a local profile")
            with st.form("first-profile"):
                name = st.text_input("Profile name", max_chars=80)
                if st.form_submit_button("Create profile", use_container_width=True):
                    try:
                        store.create_profile(name)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
            st.caption("No account connection required.")
            st.info("Create a profile, search for games you know, and add a few honest ratings.")
            return
        names = {row["id"]: row["name"] for row in profiles}
        if st.session_state.get("selected_profile") not in names:
            st.session_state["selected_profile"] = profiles[0]["id"]
        profile_id = st.selectbox("Local profile", list(names), format_func=names.get, key="selected_profile")
        page = st.radio("Navigation", ["Home", "Discover", "Search", "My Games", "Settings"], label_visibility="collapsed", key="navigation", on_change=clear_similar)
        st.divider()
        st.caption("Offline recommendations · private local profiles")
    feedback = store.feedback(profile_id)
    if page == "Settings":
        settings(store, profile_id)
        return
    if st.session_state.get("similar_seed"):
        similar_panel(store, profile_id, st.session_state["similar_seed"])
        return
    if page == "Search":
        st.title("Find a game you know")
        st.write("Rate enjoyment separately from whether you own, finished, or abandoned a game.")
        query = st.text_input("Search your local game catalog", placeholder="A title, franchise, or alternate title…")
        if query:
            matches = store.search(query, limit=30)
            if not matches:
                st.info("No local matches. Add metadata through the acquisition pipeline or import a canonical game ID with a title.")
            for game in matches:
                game_card(store, profile_id, game, "search")
        else:
            st.caption(f"{store.catalog_count():,} games available locally. Start typing a title.")
        return
    if page == "My Games":
        st.title("My Games")
        st.caption("Your ratings and activity persist across app restarts.")
        filter_state = st.selectbox("Activity filter", ["All", *STATUSES])
        rows = [row for row in feedback if filter_state == "All" or row["status"] == filter_state]
        if not rows:
            st.info("No records here yet. Search for games to rate, or import a library in Settings.")
        for row in rows[:100]:
            game = store.game(row["game_id"])
            if game:
                game = dict(game)
                game["explanation"] = f"{RATING_LABELS.get(row['rating'], 'Not rated')} · {row['status']}" + (" · Not interested" if row["not_interested"] else "")
                game_card(store, profile_id, game, "library")
        if len(rows) > 100:
            st.caption("Showing the most recent 100 records. Use the activity filter to narrow the list.")
        return
    if page == "Home":
        st.title(f"Find your next favorite, {names[profile_id]}")
        st.write("Games selected from your local catalog using population evidence and your own preferences.")
        left, middle, right = st.columns(3)
        left.metric("Games rated", sum(row["rating"] is not None for row in feedback))
        middle.metric("Games in your collection", len(feedback))
        right.metric("Imported services", len(store.profile_connections(profile_id)))
        if feedback:
            with st.expander("Recently updated"):
                for row in feedback[:5]:
                    st.write(f"{row['title']} — {RATING_LABELS.get(row['rating'], 'Not rated')} · {row['status']}")
        st.subheader("Recommended for you" if any(row["rating"] not in (None, 0) for row in feedback) else "Start exploring")
        if not any(row["rating"] not in (None, 0) for row in feedback):
            if store.implicit(profile_id):
                st.info("No explicit enjoyment ratings yet. Imported ownership/playtime can provide weak personalization when a validated implicit-history model is available. Read each recommendation’s explanation; rate familiar games for stronger preference evidence.")
            else:
                st.info("No positive or negative enjoyment ratings yet. These are population recommendations, not a personalized prediction of your enjoyment. Rate a few familiar games in Search to personalize.")
        games = recommendations(store, profile_id)
    else:
        st.title("Discover")
        filters = discover_filters(store)
        exclude_played = st.checkbox("Exclude previously played games", value=True)
        with st.expander('More discovery options'):
            limit = st.selectbox("Results", [10, 20, 40], index=1)
            content_explore = st.checkbox("Explore by game descriptions (includes games without historical interactions)", value=False)
            sort_order = st.selectbox('Sort displayed recommendations', ['Recommendation rank','Title'])
            st.caption("Content exploration can include titles with limited player history; uncertainty is higher. Ranking scores are not probabilities of enjoyment.")
        games = recommendations(store, profile_id, limit=limit, exclude_played=exclude_played, content_explore=content_explore, filters=filters)
        if sort_order == 'Title':
            games = sorted(games,key=lambda game:game['title'].casefold())
    if not games:
        st.info("No recommendations available for these settings. Add ratings, check the model in Settings, or broaden your filters.")
    for index, game in enumerate(games, 1):
        game_card(store, profile_id, game, page.lower(), rank=index)


if __name__ == "__main__":
    main()
