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

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.sigmoid(outputs.logits)

        for prob_row in probs:
            row = []
            for label, prob in zip(label_order, prob_row.tolist()):
                threshold = thresholds.get(label, 0.5)
                if prob > threshold:
                    row.append((label, prob))
            predictions.append(row)
    return predictions


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


def summarize_product(name, records, predictions, top_emotions=3):
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

    return {
        "product": name,
        "total_reviews": len(records),
        "reviews_with_text": total_texts,
        "avg_stars": avg_stars,
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

    for path in files:
        records = load_reviews(path)
        if args.max_reviews is not None:
            records = records[: args.max_reviews]

        texts = [r["text"] for r in records if r["text"]]
        predictions = predict_emotions(
            texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            thresholds=thresholds,
            batch_size=args.batch_size,
        )
        name = product_name_from_path(path)
        summaries.append(
            summarize_product(name, records, predictions, args.top_emotions)
        )

    summaries.sort(key=lambda x: (x["avg_stars"] is None, -(x["avg_stars"] or 0.0)))
    print_report(summaries)


if __name__ == "__main__":
    main()
