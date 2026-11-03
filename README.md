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
| Real data downloaded and processed (78,180 videos) | Done |
| EDA and feature notebooks | Done (run on real data) |
| Models for log(views) and engagement, interpretation by category | Next, see [Next steps](#next-steps) |

## Data

[YouTube Trending Video Dataset](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset)
by rsrishav, collected with the YouTube Data API. About 200 trending videos per day per
country since August 2020. CC0 license; public metadata only.

The data covers 2020-08-12 to 2024-04-15 (the Kaggle dataset stopped updating then).
Download with a Kaggle API token in `~/.kaggle/kaggle.json`:

```bash
for f in {US,GB,CA}_youtube_trending_data.csv {US,GB,CA}_category_id.json; do
  uvx --with six kaggle datasets download rsrishav/youtube-trending-video-dataset -f $f -p data/raw
done
cd data/raw && for z in *.zip; do unzip -o "$z" && rm "$z"; done
```

This gives these six files in `data/raw/`:

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

**English only.** The clickbait word list, VADER, and TF-IDF assume English. A title is
dropped only with real evidence it isn't English: it's mostly in a non-Latin script, or the
language detector (lingua) says non-English *and* the title contains common French/Spanish
words ("le", "de", "j'ai", "oficial") or at least 20% of its words carry accents.

This took three tries, each checked by reading random samples of dropped titles:

1. *English confidence ≥ 0.5* dropped most name-heavy English titles
   ("MrBeast | $1 vs $1,000,000 Hotel").
2. *Detector's top guess is English* still dropped about a third of English titles on real
   data, because short titles in caps confuse it: "KSI TALENT SHOW" came out as Tagalog,
   "SIDEMEN RAGE AT IMPOSSIBLE FALL GUYS MAP" as French.
3. *Current rule:* drops 4,050 of 82,230 videos (3,111 trended in CA, mostly Québec
   French). About 29 of 30 sampled drops are truly non-English, and under 1% of kept
   titles are not English (mostly K-pop titles mixing English and Korean).

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
Compare") only began rolling out in July 2025, after this dataset ends (April 2024), so
A/B variants can't appear. Result: 2.9% of videos (about 2,300) trended under more than one
title.

## Next steps

### 1. Baseline models for log(views)

Each model gets the same controls: `log_age_days`, category, `publish_week`,
`publish_hour_utc`, `publish_dow_utc`, `n_tags`, `desc_len`, `n_countries`.

| Model | Title features | Algorithm |
|-------|----------------|-----------|
| Controls only | none | Ridge (floor every title model must beat) |
| (a) Rules | 14 clickbait features + VADER + RoBERTa | OLS (readable coefficients) and Ridge |
| (b) TF-IDF | 21k word and phrase weights | Ridge, alpha by cross-validation |
| (c) Embeddings | 384-dim MiniLM vectors | Ridge, then LightGBM if Ridge is weak |

Report RMSE and R² on held-out data for **both** splits:

- `split_group` (creators never seen in training): does the title help predict views for a
  new creator?
- `split_time` (same creators, later videos), with the channel baseline added: does the
  title help beyond what that creator usually gets?

The gain over "controls only" is the title's contribution. Expect it to be small;
how small is a result in itself.

### 2. Which tactics go with views, and which are overused

- OLS on rule features with channel-clustered standard errors. Report effect sizes
  ("one extra `!` goes with X% more or fewer views") and confidence intervals.
- Same model with **channel fixed effects** (within-creator). Compare coefficients with the
  pooled model: an effect that shrinks or flips sign was about which channels use the tactic,
  not the tactic itself. This answers the instructor's pooled vs. single-creator question.
- Top positive and negative TF-IDF terms from the Ridge model, controls included.
- "Overused" = common tactics (high usage rate) whose effect is near zero or negative.

### 3. Engagement

Repeat step 2 with `log_likes_per_view` and `log_comments_per_view` as targets. Rows where
the creator disabled likes or comments are excluded, not treated as zero. Key question: do
tactics that raise views also lower likes or comments per view (the clickbait "backlash")?

### 4. By category

Fit the rule-feature models per category (top ~10 categories by count; small ones pooled
into "Other"), or add feature × category interactions. Report where tactics work and where
they backfire, e.g. caps in Gaming vs. News & Politics.

### 5. Robustness checks

- Title changes: rerun on videos with one title only (drops 2.9%) and compare.
- Country: fit US-only, GB-only, CA-only.
- Language filter: rerun with every title kept and confirm conclusions hold.
- Sentiment: VADER vs. RoBERTa (correlation is only 0.48, so they measure different things).

### 6. Stretch goals

- Fine-tune a small transformer (e.g. DistilBERT) to predict `log_views` from the title and
  compare it with the embedding + Ridge model.
- Use the YouTube Data API to fetch subscriber counts and video length, two confounders
  this dataset doesn't have.
- Interactive tool: paste a title, see its clickbait features and how similar titles did.

## Limitations

- **Trending videos only.** Every result holds only for videos that made the trending list,
  not for YouTube as a whole.
- **Correlation, not causation.** The controls narrow other explanations but can't rule them
  out.
- **English only.** Non-English titles are dropped; the count is logged by country.
