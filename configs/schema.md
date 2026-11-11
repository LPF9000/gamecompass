# Shared schema v1
Normalized population SQLite tables (data/population.sqlite):
- games(game_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT DEFAULT '', platforms TEXT DEFAULT '[]', genres TEXT DEFAULT '[]', themes TEXT DEFAULT '[]', metadata TEXT DEFAULT '{}'). JSON strings for list/dict columns. Steam IDs canonicalized as steam:<appid>.
- interactions(user_id TEXT, game_id TEXT, positive INTEGER CHECK IN (0,1), timestamp INTEGER, playtime REAL, PRIMARY KEY(user_id,game_id)). user_id is source-namespaced hashed source identity; no cross-source user merging.
Catalog and local profiles use same games schema in data/local.sqlite. Never mix local profiles into population training implicitly.
Recommendation artifacts: models/<version>/manifest.json and weights.npz (no pickle). IDs in manifest.game_ids; square weights optional; feature matrix may be sparse. Production pointer models/current.json.
All modules accept pathlib.Path data_dir; default GAMECOMPASS_DATA or ./data. Services own connections, UI does not write SQL.
