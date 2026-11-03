# YT ClickBait

Which clickbait title tactics actually go with more views on trending YouTube videos, and
which come with worse audience reception (fewer likes or comments per view)? Does the answer
change by category?

This repo analyzes trending video titles from the US, GB, and CA using hand-crafted clickbait
features and NLP (sentiment, TF-IDF, sentence embeddings). It measures how those features
relate to views and engagement while controlling for video age and the creator.

**Intended users:** YouTube creators and digital marketers choosing titles, and platform
analysts studying engagement.

## Status

| Step | State |
|------|-------|
| Repo structure, environment, tests | Done |
| Load, clean, deduplicate, language filter, train/test splits | Done |
| Rule-based features, VADER + RoBERTa sentiment, TF-IDF, embeddings | Done |
| EDA and feature notebooks | Scaffolded |
| Models for log(views) and engagement, interpretation by category | Next phase |

## Data

[YouTube Trending Video Dataset](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset)
by rsrishav, collected with the YouTube Data API. About 200 trending videos per day per
country since August 2020. CC0 license; public metadata only.

Copy these six files into `data/raw/`:

```
US_youtube_trending_data.csv   US_category_id.json
GB_youtube_trending_data.csv   GB_category_id.json
CA_youtube_trending_data.csv   CA_category_id.json
```

`data/` is gitignored. Raw and processed data are never committed.

## Setup

Needs [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 (torch and
sentence-transformers don't support 3.14 yet).

```bash
uv sync                                   # create .venv and install everything
uv run pytest -m "not slow"               # fast tests, no model downloads
uv run pytest                             # includes RoBERTa and MiniLM tests
```

## Running the pipeline

```bash
uv run python scripts/build_dataset.py    # raw CSVs -> videos.parquet + split.parquet
uv run python scripts/build_features.py   # -> rule features, sentiment, TF-IDF, embeddings
```

`build_features.py --skip-roberta --skip-embeddings` runs in seconds if you only need the
rule features and VADER. First full run downloads two models (about 600 MB) and takes several
minutes. It uses the Apple GPU (MPS) when available.

## What's in the repo

```
configs/config.yaml        All settings: paths, countries, seed, model names, TF-IDF params
data/raw/                  Kaggle files go here (gitignored)
data/processed/            Pipeline outputs (gitignored)
models/                    Saved TF-IDF vectorizer and later models (gitignored)
notebooks/                 01 EDA, 02 feature checks, 03 modeling (next phase)
reports/figures/           Saved plots
scripts/                   Command-line entry points for the pipeline
src/ytclickbait/
  config.py                Loads the YAML and resolves paths
  data/load.py             Reads country CSVs and category JSONs
  data/clean.py            One row per video, targets, and confounder controls
  data/language.py         English detection (lingua)
  data/split.py            Train/test splits and channel baselines
  features/rules.py        Hand-crafted clickbait features
  features/sentiment.py    VADER and RoBERTa sentiment
  features/tfidf.py        TF-IDF, fit on train rows only
  features/embeddings.py   all-MiniLM-L6-v2 sentence embeddings
  resources/clickbait_words.txt   Editable clickbait phrase list
  models/                  Next phase
tests/                     pytest suite on small synthetic data (no real data needed)
```

### Outputs (`data/processed/`)

All tables share the same `video_id` row order.

| File | Contents |
|------|----------|
| `videos.parquet` | One row per video: title, channel, category, targets, controls, title history |
| `split.parquet` | `split_group`, `split_time`, and a channel baseline for each split |
| `rule_features.parquet` | 14 clickbait features per title |
| `sentiment.parquet` | VADER (compound/pos/neu/neg) and RoBERTa (neg/neu/pos probabilities, pos − neg) |
| `tfidf.npz` | Sparse TF-IDF matrix (word 1–2 grams); vectorizer in `models/` |
| `embeddings.npy` | 384-dim normalized title embeddings |

## Method decisions

**One row per video.** A video appears once for each day it trended, in each country. We keep
the latest snapshot (highest views) and record `countries`, `n_countries`, `days_trending`,
and `first_trending_date`.

**Targets.** `log_views`; `likes_per_view` and `comments_per_view` (plus log versions).
Engagement is left empty, not zero, when the creator turned likes or comments off.

**Video age.** Days from publish to the snapshot, at least 1. Models use `log_age_days`.

**English only.** The clickbait word list, VADER, and TF-IDF assume English. A title is kept
when English is the most likely language among 20 languages common on these charts. Very
short titles (emoji only, for example) are kept. A fixed confidence cutoff was tried and
rejected: short, name-heavy English titles like "MrBeast | $1 vs $1,000,000 Hotel" score low
English confidence and were wrongly dropped.

**Clickbait features.** Length, caps ratio, all-caps words, `!` and `?` counts, numbers,
starting with a number, emojis, brackets, `|` or `:`, hashtags, "you/your", and clickbait
phrase hits. Generic words like "official", "vs", and "breaking" are left out of the phrase
list because they mostly mark music videos, sports, and news. TF-IDF still sees them.

**Sentiment.** VADER is the main, interpretable score; it reacts to caps and `!`, which
matters for clickbait. Twitter-trained RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
is a stronger comparison.

**No leakage.** TF-IDF vocabulary is fit only on rows that are train under both splits.
Channel baselines use train rows only. Embedding caches are tied to a fingerprint of the
titles, so a rebuilt dataset can't reuse stale vectors.

## Separating the title's effect from everything else

A video's views depend mostly on things other than its title: the creator's audience,
category, timing, thumbnail, and outside promotion. Large channels also tend to share a
title style, so a naive model would credit the title for what is really channel size. The
pipeline sets up these controls:

- **Channel baseline.** Each video's channel's typical `log_views` from train videos only.
  Train rows use an out-of-fold mean (other folds of the same channel). A leave-one-out mean
  was rejected because it leaks the row's own views: it falls exactly as that video's views
  rise. `log_views - baseline` reads as "views compared to that creator's usual." It only
  carries information for `split_time`; under `split_group` test channels are never in train.
- **Two splits.** `split_group` keeps every channel entirely in train or test, which tests
  whether the model works for creators it has never seen. `split_time` uses only channels
  with at least 10 videos and tests on each channel's latest 20%, which holds the creator
  fixed (like a per-creator model). Comparing feature effects across the two shows
  confounding: an effect that flips sign within channels was really about which channels
  use it.
- **Time and context controls.** `publish_week` (news cycles, holidays, platform changes),
  publish hour and weekday (UTC; creator time zone isn't in the data), category, tag count, description length, number of countries.
- **Not available in the data.** Subscriber count, thumbnail, video length, outside
  promotion. The channel baseline partly stands in for subscriber count.

## Title changes

The clean step records `n_distinct_titles`, `title_changed`, `first_title`, and `last_title`
for each video, using the daily snapshots across countries. This catches manual title edits
during trending and translated per-country titles. YouTube's title A/B testing ("Test &
Compare") only began rolling out in July 2025, and the API returns only the main title, so
A/B variants don't appear in this data.

## Limitations

- **Trending videos only.** Every result holds only for videos that made the trending list,
  not for YouTube as a whole.
- **Correlation, not causation.** The controls narrow other explanations but can't rule them
  out.
- **English only.** Non-English titles are dropped; the count is logged by country.
