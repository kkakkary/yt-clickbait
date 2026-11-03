# CLAUDE.md

Course project: which clickbait title features go with views and engagement on trending
YouTube videos (Kaggle rsrishav dataset, US/GB/CA). See README.md for method and decisions.

## Commands

```bash
uv sync
uv run pytest -m "not slow"     # ~2s
uv run pytest                   # ~1 min, runs RoBERTa + MiniLM
uv run ruff check . && uv run ruff format .
uv run python scripts/build_dataset.py
uv run python scripts/build_features.py [--skip-roberta] [--skip-embeddings]
```

## Gotchas

- **iCloud.** The project sits in an iCloud-synced Documents folder. iCloud evicts files
  ("dataless") and Python then hangs at 0% CPU reading them. The venv lives in
  `.venv.nosync/` (iCloud skips `*.nosync`) with `.venv` as a symlink to it. Don't recreate
  `.venv` as a real folder. Large files in `data/` can be evicted too.
- Python pinned to 3.12; torch / sentence-transformers lag behind 3.14.
- Language filter: a fixed English-confidence cutoff wrongly drops name-heavy English titles.
  Use top-language-among-candidates (`data/language.py`).
- Clean step uses `drop_duplicates(keep="last")`, not `groupby().last()`, which mixes
  non-null values from different snapshots.
- TF-IDF vocabulary and channel baselines must be fit on train rows only.
- All processed tables are row-aligned by `video_id`; build_features checks this against
  split.parquet.

## Decisions made with the user

VADER + RoBERTa sentiment; all-MiniLM-L6-v2 embeddings; drop non-English; latest snapshot
per video; channel-grouped split (plus within-channel time split, added after instructor
feedback about separating title effect from creator); src package + thin notebooks.
