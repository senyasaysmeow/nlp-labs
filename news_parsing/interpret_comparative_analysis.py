import argparse
from pathlib import Path

import pandas as pd


def pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (part / total) * 100.0


def load_required(input_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "summary": input_dir / "summary.csv",
        "similarity": input_dir / "source_similarity.csv",
        "tfidf": input_dir / "tfidf_keywords.csv",
        "lemmas": input_dir / "top_lemmas.csv",
        "pos": input_dir / "pos_distribution.csv",
        "sentiment_per_article": input_dir / "sentiment_per_article.csv",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files in input dir: " + ", ".join(missing))

    return {
        "summary": pd.read_csv(required["summary"]),
        "similarity": pd.read_csv(required["similarity"]),
        "tfidf": pd.read_csv(required["tfidf"]),
        "lemmas": pd.read_csv(required["lemmas"]),
        "pos": pd.read_csv(required["pos"]),
        "sentiment_per_article": pd.read_csv(required["sentiment_per_article"]),
    }


def top_terms(tfidf_df: pd.DataFrame, source: str, n: int = 5) -> list[str]:
    subset = tfidf_df[tfidf_df["source"] == source].sort_values(
        "tfidf", ascending=False
    )
    terms = subset["term"].astype(str).head(n).tolist()
    return terms


def top_lemmas(lemmas_df: pd.DataFrame, source: str, n: int = 5) -> list[str]:
    subset = lemmas_df[lemmas_df["source"] == source].sort_values(
        "count", ascending=False
    )
    words = subset["lemma"].astype(str).head(n).tolist()
    return words


def pos_profile(pos_df: pd.DataFrame, source: str) -> tuple[float, float, float]:
    subset = pos_df[pos_df["source"] == source].copy()
    total = float(subset["count"].sum())
    noun = float(subset[subset["pos"] == "NOUN"]["count"].sum())
    verb = float(subset[subset["pos"] == "VERB"]["count"].sum())
    adj = float(subset[subset["pos"] == "ADJF"]["count"].sum())
    return pct(noun, total), pct(verb, total), pct(adj, total)


def article_sentiment_breakdown(
    sentiment_df: pd.DataFrame, source: str
) -> tuple[float, float, float]:
    subset = sentiment_df[sentiment_df["source"] == source].copy()
    total = float(len(subset))
    pos_share = pct(float((subset["sentiment_label"] == "positive").sum()), total)
    neu_share = pct(float((subset["sentiment_label"] == "neutral").sum()), total)
    neg_share = pct(float((subset["sentiment_label"] == "negative").sum()), total)
    return pos_share, neu_share, neg_share


def build_report(data: dict[str, pd.DataFrame]) -> str:
    summary = data["summary"].copy()
    similarity = data["similarity"].copy()
    tfidf = data["tfidf"].copy()
    lemmas = data["lemmas"].copy()
    pos = data["pos"].copy()
    sentiment_per_article = data["sentiment_per_article"].copy()

    summary_sorted_ttr = summary.sort_values("ttr_lemma", ascending=False)
    summary_sorted_vocab = summary.sort_values("unique_lemmas", ascending=False)
    summary_sorted_sent = summary.sort_values("sent_mean", ascending=False)

    most_var = summary_sorted_ttr.iloc[0]
    least_var = summary_sorted_ttr.iloc[-1]
    biggest_vocab = summary_sorted_vocab.iloc[0]
    most_pos = summary_sorted_sent.iloc[0]
    most_neg = summary_sorted_sent.iloc[-1]

    sim_hi = similarity.sort_values("cosine_similarity", ascending=False).iloc[0]
    sim_lo = similarity.sort_values("cosine_similarity", ascending=True).iloc[0]

    lines = []
    lines.append("# Інтерпретація порівняльного аналізу новин")
    lines.append("")
    lines.append("## 1) Загальна картина корпусу")
    lines.append(
        f"- Найбільший словник лем: **{biggest_vocab['source']}** ({int(biggest_vocab['unique_lemmas'])} унікальних лем)."
    )
    lines.append(
        f"- Найвища лексична різноманітність (TTR): **{most_var['source']}** ({most_var['ttr_lemma']:.4f})."
    )
    lines.append(
        f"- Найнижча лексична різноманітність (TTR): **{least_var['source']}** ({least_var['ttr_lemma']:.4f})."
    )
    lines.append(
        "- Інтерпретація: вищий TTR зазвичай означає ширший тематичний діапазон або менше повторів шаблонних конструкцій."
    )
    lines.append("")
    lines.append("## 2) Подібність джерел (TF-IDF + cosine)")
    lines.append(
        f"- Найбільш схожі: **{sim_hi['source_a']} ↔ {sim_hi['source_b']}** (cosine={sim_hi['cosine_similarity']:.3f})."
    )
    lines.append(
        f"- Найменш схожі: **{sim_lo['source_a']} ↔ {sim_lo['source_b']}** (cosine={sim_lo['cosine_similarity']:.3f})."
    )
    lines.append(
        "- Інтерпретація: вища схожість означає ближчий порядок денний, спільні теми та подібний словник."
    )
    lines.append("")
    lines.append("## 3) Тональність (лексиконний підхід)")
    lines.append(
        f"- Найбільш позитивний середній індекс: **{most_pos['source']}** (mean={most_pos['sent_mean']:.4f})."
    )
    lines.append(
        f"- Найбільш негативний середній індекс: **{most_neg['source']}** (mean={most_neg['sent_mean']:.4f})."
    )
    lines.append(
        "- Висновок: усі індекси близькі до нуля, тому стрічки загалом близькі до нейтральних."
    )

    for _, row in summary.iterrows():
        src = row["source"]
        lines.append(
            f"- {src}: pos_share={row['sent_pos_share']:.3f}, neg_share={row['sent_neg_share']:.3f}."
        )

    lines.append(
        "- Додатково доступний файл на рівні новин: sentiment_per_article.csv (оцінка та мітка для кожної новини)."
    )

    lines.append("")
    lines.append("## 3.1) Тональність по кожній новині")
    for src in summary["source"].tolist():
        pos_p, neu_p, neg_p = article_sentiment_breakdown(sentiment_per_article, src)
        lines.append(
            f"- {src}: positive={pos_p:.1f}%, neutral={neu_p:.1f}%, negative={neg_p:.1f}%."
        )

    top_positive = sentiment_per_article.sort_values(
        "sentiment_score", ascending=False
    ).head(3)
    top_negative = sentiment_per_article.sort_values(
        "sentiment_score", ascending=True
    ).head(3)

    lines.append("- Найбільш позитивні матеріали (top-3 за score):")
    for _, row in top_positive.iterrows():
        title = str(row.get("title", "")).strip() or "[без заголовка]"
        lines.append(
            f"- {row['source']} | score={row['sentiment_score']:.4f} | {title}"
        )

    lines.append("- Найбільш негативні матеріали (top-3 за score):")
    for _, row in top_negative.iterrows():
        title = str(row.get("title", "")).strip() or "[без заголовка]"
        lines.append(
            f"- {row['source']} | score={row['sentiment_score']:.4f} | {title}"
        )

    lines.append("")
    lines.append("## 4) Структура мови (POS)")
    for src in summary["source"].tolist():
        noun_p, verb_p, adj_p = pos_profile(pos, src)
        lines.append(
            f"- {src}: NOUN={noun_p:.1f}%, VERB={verb_p:.1f}%, ADJF={adj_p:.1f}% (домінує фактологічний стиль із перевагою іменників)."
        )

    lines.append("")
    lines.append("## 5) Ключові маркери контенту")
    for src in summary["source"].tolist():
        terms = ", ".join(top_terms(tfidf, src, n=5))
        lemma_list = ", ".join(top_lemmas(lemmas, src, n=5))
        lines.append(f"- {src} TF-IDF маркери: {terms}.")
        lines.append(f"- {src} частотні леми: {lemma_list}.")

    lines.append("")
    lines.append("## 6) Практичний висновок")
    lines.append(
        "- Якщо ціль — максимум різноманіття тем, орієнтир: джерело з найвищим TTR."
    )
    lines.append(
        "- Якщо ціль — найближчі за порядком денним стрічки, орієнтир: пара з найбільшою cosine-схожістю."
    )
    lines.append(
        "- Якщо ціль — моніторинг тональності, краще доповнити лексиконний метод ML-моделлю sentiment для української мови."
    )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret results from comparative news NLP analysis"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("analysis") / "comparative_nlp",
        help="Directory with summary.csv, source_similarity.csv, tfidf_keywords.csv, top_lemmas.csv, pos_distribution.csv, sentiment_per_article.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis") / "comparative_nlp" / "interpretation.md",
        help="Output markdown report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_required(args.input_dir)
    report = build_report(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Interpretation saved: {args.output}")


if __name__ == "__main__":
    main()
