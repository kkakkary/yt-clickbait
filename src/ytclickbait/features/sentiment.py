"""Title sentiment: VADER (rule-based, reads caps and !) and a Twitter-trained RoBERTa."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_scores(titles: pd.Series) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = [analyzer.polarity_scores(t) for t in titles]
    df = pd.DataFrame(rows, index=titles.index)
    return df.rename(columns=lambda c: f"vader_{c}")


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def roberta_scores(
    titles: pd.Series,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 64,
    device: str | None = None,
) -> pd.DataFrame:
    """Probability of negative / neutral / positive, plus pos - neg as one score."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or best_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]

    texts = titles.tolist()
    probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa sentiment"):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())

    df = pd.DataFrame(
        np.vstack(probs), index=titles.index, columns=[f"roberta_{lb}" for lb in labels]
    )
    df["roberta_score"] = df["roberta_positive"] - df["roberta_negative"]
    return df


def sentiment_features(
    titles: pd.Series, use_roberta: bool = True, **roberta_kwargs
) -> pd.DataFrame:
    parts = [vader_scores(titles)]
    if use_roberta:
        parts.append(roberta_scores(titles, **roberta_kwargs))
    return pd.concat(parts, axis=1)
