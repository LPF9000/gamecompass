# Local application guide

Launch `gamecompass app`, then open the loopback URL it prints. The UI is for a personal computer, not a public multiuser server. Training/acquisition run as separate CLI jobs so feedback/search UI can remain responsive; do not launch multiple acquisition processes against the same data path.

Create independent profiles from the sidebar or Settings. Search familiar titles, open a game and save feedback. Loved=2, Liked=1, Neutral=0, Disliked=-1, Hated=-2. Insufficient experience leaves enjoyment unset. Not interested is an exclusion, not a dislike label. Owned, played, completed, abandoned, wishlist and unplayed are independent activity states. Optional reasons are retained; their separate predictive effect is not yet validated.

Discover excludes judged/unwanted games and can exclude played/completed/abandoned games. Platform filters use available metadata only. Content exploration includes games without collaborative history; uncertainty is higher for these games. Explanations describe actual input features/model route and do not infer psychological traits.

My Games lists local feedback and imported history. Steam libraries add independent playtime/ownership signals; explicit ratings are preserved when reimporting. Incomplete/private library responses are reported rather than treated as empty success. Recent activity and single-game achievements can be viewed on demand.

Settings also provides a missing-game/new-release form. Supply a verified canonical service ID, description and platforms; these user-supplied fields become available to search and content exploration immediately, without population retraining. Existing canonical records require an explicit update selection.

Manual CSV import accepts `game_id,title,rating,status,not_interested,playtime_minutes`. Use canonical IDs such as `steam:12345`; a supplied title cannot force a fuzzy identity merge. Blank enjoyment means unjudged. JSON profile exports preserve profile data, feedback and outcome/evaluation records; imports create an independent profile and validate the payload transactionally. Duplicate canonical IDs update the same record. Credentials are never exported.

Personal evaluation in Settings temporarily withholds known feedback for scoring without changing saved preferences. It is distinct from the relevance feedback shown alongside recommendations and from post-play enjoyment. Results are local and can be inspected over time. Deleting a profile removes its associated database records; independently saved exports remain separate.

Model/data diagnostics show local artifacts and errors. Use previous intact model versions for rollback. No automatic training, scheduled jobs or data sharing are enabled.
