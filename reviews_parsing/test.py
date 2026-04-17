import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "ukr-detect/emotions_classifier"
DEFAULT_THRESHOLDS = {
    "Joy": 0.35,
    "Fear": 0.5,
    "Anger": 0.25,
    "Sadness": 0.5,
    "Disgust": 0.3,
    "Surprise": 0.25,
    "None": 0.35,
}

POSITIVE_LABELS = {"Joy", "Surprise"}
NEGATIVE_LABELS = {"Fear", "Anger", "Sadness", "Disgust"}
NEUTRAL_LABEL = "None"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        return torch.device("mps")
    print("No GPU available, using CPU.")
    return torch.device("cpu")


def init_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = pick_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_emotions(texts, tokenizer, model, device, thresholds, batch_size=32):
    id2label = model.config.id2label
    label_order = [id2label[i] for i in range(len(id2label))]

    thresholded_predictions = []
    probability_rows = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.sigmoid(outputs.logits)

        for prob_row in probs:
            prob_dict = {
                label: prob for label, prob in zip(label_order, prob_row.tolist())
            }
            probability_rows.append(prob_dict)

            row = []
            for label, prob in prob_dict.items():
                threshold = thresholds.get(label, 0.5)
                if prob > threshold:
                    row.append((label, prob))
            thresholded_predictions.append(row)

    return thresholded_predictions, probability_rows


def tonal_metrics_from_probs(prob_dict):
    pos = sum(prob_dict.get(label, 0.0) for label in POSITIVE_LABELS)
    neg = sum(prob_dict.get(label, 0.0) for label in NEGATIVE_LABELS)
    neu = prob_dict.get(NEUTRAL_LABEL, 0.0)
    total = pos + neg + neu

    if total <= 0:
        return {
            "tonal_label": "Neutral",
            "tonal_score": 0.0,
            "tonal_stars": 3.0,
            "confidence": 0.0,
        }

    buckets = {
        "Positive": pos / total,
        "Negative": neg / total,
        "Neutral": neu / total,
    }
    tonal_label = max(buckets.items(), key=lambda x: x[1])[0]
    confidence = buckets[tonal_label]
    tonal_score = (pos - neg) / total
    tonal_stars = max(1.0, min(5.0, 3.0 + 2.0 * tonal_score))

    return {
        "tonal_label": tonal_label,
        "tonal_score": tonal_score,
        "tonal_stars": tonal_stars,
        "confidence": confidence,
    }


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "product"


def write_product_output(output_dir, product_name, records, tonal_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"product_analysis_{safe_name(product_name)}.json"

    payload = []
    for record, tonal in zip(records, tonal_rows):
        payload.append(
            {
                "review": record["text"],
                "stars": record["stars"],
                "tonal_label": tonal["tonal_label"],
                "tonal_score": round(tonal["tonal_score"], 4),
                "tonal_stars": round(tonal["tonal_stars"], 4),
                "confidence": round(tonal["confidence"], 4),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def product_name_from_path(path: Path) -> str:
    m = re.search(r"makeup_reviews\((.+)\)\.json$", path.name)
    if m:
        return m.group(1)
    return path.stem


def load_reviews(path: Path):
    with path.open("r", encoding="utf-8") as f:
        docs = json.load(f)

    records = []
    for doc in docs:
        text = (doc.get("text") or "").strip()
        stars = doc.get("stars")
        records.append({"text": text, "stars": stars})
    return records


def summarize_product(name, records, predictions, tonal_rows, top_emotions=3):
    valid_texts = [r["text"] for r in records if r["text"]]
    rated_stars = [
        r["stars"]
        for r in records
        if isinstance(r["stars"], (int, float)) and r["stars"] > 0
    ]

    emotion_counts = Counter()
    for pred in predictions:
        for label, _ in pred:
            emotion_counts[label] += 1

    total_texts = len(valid_texts)
    emotion_share = []
    if total_texts > 0:
        for label, count in emotion_counts.items():
            emotion_share.append((label, count / total_texts))
        emotion_share.sort(key=lambda x: x[1], reverse=True)

    top = emotion_share[:top_emotions]
    avg_stars = sum(rated_stars) / len(rated_stars) if rated_stars else None
    avg_tonal_stars = (
        sum(row["tonal_stars"] for row in tonal_rows) / len(tonal_rows)
        if tonal_rows
        else None
    )
    avg_tonal_score = (
        sum(row["tonal_score"] for row in tonal_rows) / len(tonal_rows)
        if tonal_rows
        else None
    )
    stars_gap = (
        avg_tonal_stars - avg_stars
        if avg_tonal_stars is not None and avg_stars is not None
        else None
    )

    return {
        "product": name,
        "total_reviews": len(records),
        "reviews_with_text": total_texts,
        "avg_stars": avg_stars,
        "avg_tonal_stars": avg_tonal_stars,
        "avg_tonal_score": avg_tonal_score,
        "stars_gap": stars_gap,
        "top_emotions": top,
        "emotion_counts": emotion_counts,
    }


def print_report(summaries):
    print("\n=== Comparative product analysis ===")
    for s in summaries:
        print(f"\nProduct: {s['product']}")
        print(f"  Reviews total: {s['total_reviews']}")
        print(f"  Reviews with text: {s['reviews_with_text']}")
        if s["avg_stars"] is None:
            print("  Avg stars: n/a")
        else:
            print(f"  Avg stars: {s['avg_stars']:.2f}")

        if s["avg_tonal_stars"] is None:
            print("  Avg tonal stars (model): n/a")
        else:
            print(f"  Avg tonal stars (model): {s['avg_tonal_stars']:.2f}")

        if s["avg_tonal_score"] is None:
            print("  Avg tonal score [-1..1]: n/a")
        else:
            print(f"  Avg tonal score [-1..1]: {s['avg_tonal_score']:.3f}")

        if s["stars_gap"] is None:
            print("  Stars gap (model - reviews): n/a")
        else:
            print(f"  Stars gap (model - reviews): {s['stars_gap']:+.2f}")

        if not s["top_emotions"]:
            print("  Top emotions: n/a")
        else:
            pairs = [
                f"{label} {share * 100:.1f}%" for label, share in s["top_emotions"]
            ]
            print(f"  Top emotions: {', '.join(pairs)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare emotions and ratings across multiple makeup products."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON files with reviews. If skipped, script auto-loads makeup_reviews(*).json from current directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Limit reviews per product.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--top-emotions",
        type=int,
        default=3,
        help="How many top emotions to print per product.",
    )
    parser.add_argument(
        "--output-dir",
        default="product_outputs",
        help="Directory for per-product JSON outputs.",
    )
    return parser.parse_args()


def resolve_files(input_files):
    if input_files:
        return [Path(p) for p in input_files]
    return sorted(Path(".").glob("makeup_reviews(*).json"))


def main():
    args = parse_args()
    files = resolve_files(args.files)

    if not files:
        raise SystemExit("No review JSON files found.")

    tokenizer, model, device = init_model(args.model)
    thresholds = DEFAULT_THRESHOLDS
    summaries = []
    output_paths = []

    for path in files:
        records = load_reviews(path)
        if args.max_reviews is not None:
            records = records[: args.max_reviews]

        text_records = [r for r in records if r["text"]]
        texts = [r["text"] for r in text_records]

        predictions, probability_rows = predict_emotions(
            texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            thresholds=thresholds,
            batch_size=args.batch_size,
        )
        tonal_rows = [
            tonal_metrics_from_probs(prob_row) for prob_row in probability_rows
        ]

        name = product_name_from_path(path)
        output_path = write_product_output(
            output_dir=Path(args.output_dir),
            product_name=name,
            records=text_records,
            tonal_rows=tonal_rows,
        )
        output_paths.append(output_path)

        summaries.append(
            summarize_product(name, records, predictions, tonal_rows, args.top_emotions)
        )

    summaries.sort(key=lambda x: (x["avg_stars"] is None, -(x["avg_stars"] or 0.0)))
    print_report(summaries)

    print("\nPer-product output files:")
    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
