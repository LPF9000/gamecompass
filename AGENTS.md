# Project instructions
Read PLAN.md and docs/STATUS.md before work. Update evidence and unresolved requirements as work progresses.
Never fabricate metrics or use synthetic training data as a replacement for real data. Fixtures are allowed only in tests.
Keep datasets, credentials, user histories and large artifacts out of Git. Check third-party licenses before distributing data or weights.
Preserve enjoyment and activity separately. Explicit negatives must not become positives; unknown feedback is not dislike.
Cold-start evaluation must hold entire users out of population training. Tune only on validation; evaluate all models on identical eligible candidates.
Use typed modular Python, SQLite transactions and parameterized SQL. Inference must work offline. Bind the UI to loopback by default.
Run pytest before commits. Document exact commands and important unverified limitations.
