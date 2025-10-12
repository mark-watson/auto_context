Files you should add to the chat (so I can edit, run, and test AutoContext effectively)

1) README.md
   - Why: Shows intended usage, examples, and expected prompt/format. Helpful to ensure any changes keep the public API and prompt formatting consistent.

2) data/ directory (all .txt files)
   - Why: The corpus is required to reproduce indexing, retrieval results, and to write realistic tests. Please add the actual files under data/ (you previously listed data/*.txt).

3) test1.py and test2.py or a tests/ directory
   - Why: Existing tests or test scripts let me run the current behavior and confirm refactors don't break functionality. If you don't have tests yet, adding basic unit tests would be very helpful.

4) .gitignore and uv.lock (optional but useful)
   - Why: Helpful for recommending new files and for knowing lockfile state / exact dependency versions.

5) Any CLI or wrapper scripts (e.g., run_autocontext.py)
   - Why: If you have an example runner, adding it makes it straightforward to verify end-to-end behavior.

6) CI configuration (optional): .github/workflows/*, tox.ini, or similar
   - Why: If present, it helps ensure changes remain compatible with automated checks.

What I'll do once you add these:
- Inspect README and tests to confirm expected behavior.
- Run or write tests exercising chunking, BM25 + vector hybrid retrieval, and get_prompt formatting.
- If requested, implement changes such as persisting indexes, batching embeddings, configurable chunking, or improved error handling.

Suggested commands you can run locally (from repo root) after adding files:
- git status
- python -c "from auto_context import AutoContext; print('AutoContext loaded:', hasattr(__import__('auto_context'), 'AutoContext'))"
- pip install -e .

If you add the files, tell me which change you'd like first (e.g., persist indexes, expose chunk-size config, add unit tests, or optimize embedding batching) and I will prepare precise SEARCH/REPLACE edits.
