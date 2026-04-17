import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show top positive and negative reviews for each product."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Per-product analysis JSON files. If empty, script reads product_analysis_*.json from --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default="product_outputs",
        help="Directory with product_analysis_*.json files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many positive and negative reviews to show per product.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore reviews below this confidence threshold.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=180,
        help="Max review length in output.",
    )
    return parser.parse_args()


def resolve_files(files, input_dir):
    if files:
        return [Path(p) for p in files]
    return sorted(Path(input_dir).glob("product_analysis_*.json"))


def product_name_from_path(path: Path):
    m = re.search(r"product_analysis_(.+)\.json$", path.name)
    if m:
        return m.group(1)
    return path.stem


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        review = (item.get("review") or "").strip()
        tonal_score = item.get("tonal_score")
        confidence = item.get("confidence")
        if not review:
            continue
        if not isinstance(tonal_score, (int, float)):
            continue
        if not isinstance(confidence, (int, float)):
            continue
        rows.append(item)
    return rows


def compact_text(text, max_chars):
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 1] + "…"


def print_block(title, rows, max_chars):
    print(f"  {title}")
    if not rows:
        print("    - n/a")
        return
    for idx, row in enumerate(rows, start=1):
        score = row.get("tonal_score", 0.0)
        confidence = row.get("confidence", 0.0)
        stars = row.get("tonal_stars")
        text = compact_text(row.get("review", ""), max_chars)
        if isinstance(stars, (int, float)):
            print(
                f"    {idx}. score={score:+.3f} conf={confidence:.3f} model_stars={stars:.2f} | {text}"
            )
        else:
            print(f"    {idx}. score={score:+.3f} conf={confidence:.3f} | {text}")


def main():
    args = parse_args()
    files = resolve_files(args.files, args.input_dir)
    if not files:
        raise SystemExit("No product analysis JSON files found.")
    print("\n=== Top positive/negative reviews by product ===")
    for path in files:
        rows = load_rows(path)
        rows = [r for r in rows if r.get("confidence", 0.0) >= args.min_confidence]
        positive = sorted(
            rows,
            key=lambda r: (r.get("tonal_score", 0.0), r.get("confidence", 0.0)),
            reverse=True,
        )[: args.top_n]
        negative = sorted(
            rows,
            key=lambda r: (r.get("tonal_score", 0.0), -r.get("confidence", 0.0)),
        )[: args.top_n]
        name = product_name_from_path(path)
        print(f"\nProduct: {name}")
        print_block("Top positive:", positive, args.max_chars)
        print_block("Top negative:", negative, args.max_chars)


if __name__ == "__main__":
    main()
