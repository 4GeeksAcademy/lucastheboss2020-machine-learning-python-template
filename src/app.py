"""
Production inference path for WeLoveReviews sentiment analysis.

Reads the raw reviews, scores every review with a pinned pretrained
sentiment model (nlptown/bert-base-multilingual-uncased-sentiment),
maps predictions to sentiment bands, and writes the enriched output to:

    data/processed/reviews_with_sentiment.csv

This mirrors the analysis logic from src/explore.ipynb, consolidated so it
can be re-run for future review drops.
"""

import os

import pandas as pd
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # pinned

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "reviews.csv"
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed",
    "reviews_with_sentiment.csv",
)


def stars_to_band(stars: int) -> str:
    """Map the model's 1-5 star output into a sentiment band."""
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"


def parse_label(label: str) -> int:
    """Parse a pipeline label like '5 stars' into the integer 5."""
    return int(label.split()[0])


def load_model():
    """Load the model ONCE, outside any per-review loop."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
    )


def main():
    # 1. Load raw reviews
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} reviews from {RAW_PATH}")

    # 2. Load the sentiment model once
    sentiment_pipeline = load_model()

    # 3. Score all reviews in one batched pass
    raw_preds = sentiment_pipeline(df["review_text"].tolist(), batch_size=64)

    # 4. Parse and map to bands
    df["predicted_stars"] = [parse_label(p["label"]) for p in raw_preds]
    df["predicted_score"] = [round(p["score"], 4) for p in raw_preds]
    df["predicted_sentiment"] = df["predicted_stars"].map(stars_to_band)

    # 5. Write enriched output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} enriched reviews to {OUT_PATH}")
    print("\nSentiment breakdown:")
    print(df["predicted_sentiment"].value_counts().to_string())
    return df


if __name__ == "__main__":
    main()
