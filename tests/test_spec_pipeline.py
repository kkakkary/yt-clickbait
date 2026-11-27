"""Full-data acceptance checks. Run with scripts/test_evidence.py --pipeline."""

import importlib.util
from dataclasses import replace
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest

from ytclickbait.config import load_config
from ytclickbait.features.tfidf import load_tfidf

pytestmark = pytest.mark.full_data


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    config = load_config()
    raw_dir = config.path("raw")
    for country in config["countries"]:
        for suffix in ("youtube_trending_data.csv", "category_id.json"):
            if not (raw_dir / f"{country}_{suffix}").is_file():
                pytest.skip(f"Missing raw input: {country}_{suffix}")

    root = tmp_path_factory.mktemp("pipeline")
    paths = {**config["paths"], "raw": str(raw_dir)}
    config = replace(config, raw={**config.raw, "paths": paths}, root=root)
    config.path("models").mkdir(parents=True)
    build = load_script("build_dataset")
    build.load_config = lambda: config
    build.main()
    return config, build


def checksums(config):
    results = {}
    for name in ("videos", "split"):
        with config.file(name).open("rb") as source:
            results[name] = file_digest(source, "sha256").hexdigest()
    return results


def test_build_dataset(dataset, caplog):
    config, build = dataset
    first = checksums(config)
    first_videos = pd.read_parquet(config.file("videos"))
    with caplog.at_level("INFO", logger="build_dataset"):
        build.main()
    second = checksums(config)
    videos = pd.read_parquet(config.file("videos"))
    if second["videos"] != first["videos"]:
        changed_columns = [
            column for column in first_videos.columns
            if column not in videos or not first_videos[column].equals(videos[column])
        ]
        pytest.fail(
            f"videos.parquet differs across builds. SHA-256: "
            f"{first['videos']} -> {second['videos']}. "
            f"Columns with differences: {changed_columns}. "
            f"Column layout unchanged: {first_videos.columns.equals(videos.columns)}."
        )
    assert second["split"] == first["split"]
    assert "Non-English videos dropped:" in caplog.text
    assert "Split counts:" in caplog.text

    split = pd.read_parquet(config.file("split"))
    assert 0.99 * 78_180 <= len(videos) <= 1.01 * 78_180
    assert videos["video_id"].notna().all() and videos["video_id"].is_unique
    assert videos["video_id"].equals(split["video_id"])
    assert videos["channel_id"].notna().all()
    assert pd.api.types.is_numeric_dtype(videos["view_count"])
    assert videos["view_count"].gt(0).all()
    train = split["split_group"].eq("train")
    test = split["split_group"].eq("test")
    assert train.any() and test.any() and (train | test).all()
    assert set(videos.loc[train, "channel_id"]).isdisjoint(videos.loc[test, "channel_id"])
    print(f"videos={len(videos)} train={train.sum()} test={test.sum()} SHA256={first}")
    print(caplog.text)


def test_build_features(dataset, monkeypatch):
    config, _ = dataset
    build = load_script("build_features")
    monkeypatch.setattr(build, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["build_features", "--skip-roberta", "--skip-embeddings"])
    started = perf_counter()
    build.main()
    elapsed = perf_counter() - started
    videos = pd.read_parquet(config.file("videos"), columns=["video_id", "title"])
    ids = videos["video_id"]
    rules = pd.read_parquet(config.file("rule_features"))
    sentiment = pd.read_parquet(config.file("sentiment"))
    vectorizer, tfidf = load_tfidf(
        config.file("tfidf_matrix"), config.file("tfidf_vectorizer", "models")
    )
    assert ids.equals(rules["video_id"])
    assert ids.equals(sentiment["video_id"])
    assert tfidf.shape[0] == len(ids)
    sampled_titles = videos["title"].sample(n=30, random_state=config["seed"])
    assert (tfidf[sampled_titles.index] - vectorizer.transform(sampled_titles)).nnz == 0
    assert rules.shape[1] == 15
    feature_types = rules.drop(columns="video_id").dtypes
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in feature_types)
    assert rules["caps_ratio"].between(0, 1).all()
    assert rules.notna().all().all()

    split = pd.read_parquet(config.file("split"))
    split.iloc[::-1].to_parquet(config.file("split"), index=False)
    with pytest.raises(ValueError, match="out of sync"):
        build.main()
    print(f"Feature build: {elapsed:.2f}s; rules={rules.shape}, sentiment={sentiment.shape}, "
          f"TF-IDF={tfidf.shape}; reordered split rejected")
    assert elapsed < 120
