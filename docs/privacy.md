# Privacy and offline operation

Profiles, ratings, account imports and evaluation outcomes stay in the selected data directory. No automatic uploads, shared training, hosted inference, ad services or analytics are implemented. The launcher binds to 127.0.0.1 and disables Streamlit usage statistics. Do not expose that port publicly; this is a single-machine app without network authentication.

Steam credentials are environment variables used only for authorized requests to Steam. They are not exported, logged or saved by the application. IGDB credentials are optional environment variables, and metadata caching requires explicit acknowledgement of provider terms. Never place real keys in tracked configuration.

`local.sqlite` is created with owner-only permissions. The database is not encrypted: OS account access and disk encryption determine protection at rest. Downloaded public population data is separate from private profiles. Search/recommendation inference uses cached data locally; initial downloads and requested service imports require internet. Covers are absent by default; displaying an external cover URL can contact its provider.

JSON/CSV exports contain personal data and are created only on request. Keep them private. Deleting a profile cascades associated feedback, connections, implicit history, outcomes and personal evaluation records in the local database. Independently saved exports/backups/manual training snapshots are separate copies and must be removed independently. Deletion is not a forensic disk-erasure guarantee.

No future sharing is enabled by default. Any shared learning feature must require explicit, granular opt-in and a documented export preview. Current manual retraining uses the downloaded population dataset; immediate personalization uses local feedback without altering that dataset.
