import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pymorphy3
except ImportError:  # pragma: no cover
    pymorphy3 = None


TOKEN_RE = re.compile(r"[a-zа-щьюяєіїґ']+", re.IGNORECASE)

STEM_SUFFIXES = [
    "ування",
    "ювання",
    "ення",
    "ання",
    "ями",
    "ові",
    "еві",
    "ими",
    "ого",
    "ому",
    "ами",
    "ях",
    "ах",
    "ям",
    "ем",
    "ом",
    "ою",
    "ею",
    "ий",
    "ій",
    "а",
    "я",
    "у",
    "ю",
    "і",
    "и",
    "о",
    "е",
    "ь",
]

POSITIVE_LEMMAS = {
    "успіх",
    "перемога",
    "відновлення",
    "розвиток",
    "підтримка",
    "допомога",
    "врятувати",
    "стабільний",
    "ефективний",
    "покращення",
    "зростання",
    "безпека",
    "захист",
    "досягнення",
    "інновація",
    "мир",
    "сильний",
    "надія",
    "позитивний",
}

NEGATIVE_LEMMAS = {
    "війна",
    "обстріл",
    "атака",
    "вибух",
    "загибель",
    "поранення",
    "криза",
    "корупція",
    "загроза",
    "руйнування",
    "втрата",
    "дефіцит",
    "блокада",
    "санкція",
    "злочин",
    "аварія",
    "пожежа",
    "катастрофа",
    "небезпека",
    "тривога",
}


@dataclass
class SourceResult:
    source: str
    articles: int
    tokens_raw: int
    tokens_no_stop: int
    unique_lemmas: int
    ttr_lemma: float
    sent_mean: float
    sent_median: float
    sent_pos_share: float
    sent_neg_share: float


def load_stopwords(base_dir: Path) -> set[str]:
    stopwords = set()
    stopwords_dir = base_dir / "stopwords"
    for filename in ("stopwords_ua.txt", "stopwords_eng.txt"):
        path = stopwords_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                token = line.strip().lower()
                if token:
                    stopwords.add(token)
    return stopwords


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def stem_uk(word: str) -> str:
    for suffix in STEM_SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class MorphTools:
    def __init__(self) -> None:
        self.enabled = pymorphy3 is not None
        self._lemma_cache: dict[str, str] = {}
        self._pos_cache: dict[str, str] = {}
        self._morph = None
        if self.enabled:
            self._morph = pymorphy3.MorphAnalyzer(lang="uk")

    def lemma(self, token: str) -> str:
        if token in self._lemma_cache:
            return self._lemma_cache[token]
        if not self._morph:
            self._lemma_cache[token] = token
            return token
        parsed = self._morph.parse(token)[0]
        value = parsed.normal_form
        self._lemma_cache[token] = value
        return value

    def pos(self, lemma: str) -> str:
        if lemma in self._pos_cache:
            return self._pos_cache[lemma]
        if not self._morph:
            self._pos_cache[lemma] = "UNKN"
            return "UNKN"
        parsed = self._morph.parse(lemma)[0]
        value = parsed.tag.POS or "UNKN"
        self._pos_cache[lemma] = value
        return value


def build_source_name(path: Path) -> str:
    raw = path.stem.replace("_news", "")
    return raw.replace("_", " ").title()


def read_news(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def analyze(
    articles_dir: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    json_files = sorted(articles_dir.glob("*_news.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_news.json files found in {articles_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stopwords = load_stopwords(articles_dir.parent)
    morph = MorphTools()

    source_docs_for_tfidf: list[str] = []
    source_labels: list[str] = []
    summary_rows: list[SourceResult] = []
    pos_rows: list[dict] = []
    lemma_rows: list[dict] = []
    stem_rows: list[dict] = []
    article_sentiment_rows: list[dict] = []

    for path in json_files:
        source = build_source_name(path)
        items = read_news(path)

        docs: list[str] = []
        tokens_raw = 0
        tokens_no_stop = 0
        lemma_counter: Counter[str] = Counter()
        stem_counter: Counter[str] = Counter()
        pos_counter: Counter[str] = Counter()
        sentiment_scores: list[float] = []

        for article_idx, item in enumerate(items, start=1):
            text = f"{item.get('title', '')} {item.get('content', '')}".strip()
            docs.append(text)

            tokens = tokenize(text)
            tokens_raw += len(tokens)
            filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
            tokens_no_stop += len(filtered)

            lemmas = [morph.lemma(t) for t in filtered]
            stems = [stem_uk(l) for l in lemmas]

            lemma_counter.update(lemmas)
            stem_counter.update(stems)
            pos_counter.update(morph.pos(l) for l in lemmas)

            pos_hits = sum(1 for l in lemmas if l in POSITIVE_LEMMAS)
            neg_hits = sum(1 for l in lemmas if l in NEGATIVE_LEMMAS)
            sentiment_score = (pos_hits - neg_hits) / max(len(lemmas), 1)
            sentiment_scores.append(sentiment_score)

            if sentiment_score > 0:
                sentiment_label = "positive"
            elif sentiment_score < 0:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            article_sentiment_rows.append(
                {
                    "source": source,
                    "article_index": article_idx,
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "time": item.get("time", ""),
                    "category": item.get("category", ""),
                    "tokens_no_stop": len(lemmas),
                    "positive_hits": pos_hits,
                    "negative_hits": neg_hits,
                    "sentiment_score": round(sentiment_score, 6),
                    "sentiment_label": sentiment_label,
                }
            )

        unique_lemmas = len(lemma_counter)
        ttr = unique_lemmas / max(tokens_no_stop, 1)
        sent_mean = mean(sentiment_scores) if sentiment_scores else 0.0
        sent_median = median(sentiment_scores) if sentiment_scores else 0.0
        sent_pos_share = sum(1 for x in sentiment_scores if x > 0) / max(
            len(sentiment_scores), 1
        )
        sent_neg_share = sum(1 for x in sentiment_scores if x < 0) / max(
            len(sentiment_scores), 1
        )

        summary_rows.append(
            SourceResult(
                source=source,
                articles=len(items),
                tokens_raw=tokens_raw,
                tokens_no_stop=tokens_no_stop,
                unique_lemmas=unique_lemmas,
                ttr_lemma=round(ttr, 6),
                sent_mean=round(sent_mean, 6),
                sent_median=round(sent_median, 6),
                sent_pos_share=round(sent_pos_share, 6),
                sent_neg_share=round(sent_neg_share, 6),
            )
        )

        for lemma, count in lemma_counter.most_common(top_n):
            lemma_rows.append({"source": source, "lemma": lemma, "count": count})
        for stem, count in stem_counter.most_common(top_n):
            stem_rows.append({"source": source, "stem": stem, "count": count})
        for pos, count in pos_counter.items():
            pos_rows.append({"source": source, "pos": pos, "count": count})

        source_labels.append(source)
        source_docs_for_tfidf.append(" ".join(docs))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[a-zа-щьюяєіїґ']+",
        stop_words=sorted(stopwords),
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )
    matrix = vectorizer.fit_transform(source_docs_for_tfidf)
    features = vectorizer.get_feature_names_out()

    tfidf_rows: list[dict] = []
    for row_idx, source in enumerate(source_labels):
        row = matrix[row_idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        for idx in top_idx:
            score = float(row[idx])
            if score <= 0:
                continue
            tfidf_rows.append(
                {
                    "source": source,
                    "term": features[idx],
                    "tfidf": round(score, 6),
                }
            )

    similarity = cosine_similarity(matrix)
    sim_rows: list[dict] = []
    for i, src_a in enumerate(source_labels):
        for j, src_b in enumerate(source_labels):
            if j <= i:
                continue
            sim_rows.append(
                {
                    "source_a": src_a,
                    "source_b": src_b,
                    "cosine_similarity": round(float(similarity[i, j]), 6),
                }
            )

    pd.DataFrame([row.__dict__ for row in summary_rows]).to_csv(
        output_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(lemma_rows).to_csv(
        output_dir / "top_lemmas.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stem_rows).to_csv(
        output_dir / "top_stems.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pos_rows).to_csv(
        output_dir / "pos_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(tfidf_rows).to_csv(
        output_dir / "tfidf_keywords.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sim_rows).to_csv(
        output_dir / "source_similarity.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(article_sentiment_rows).to_csv(
        output_dir / "sentiment_per_article.csv", index=False, encoding="utf-8-sig"
    )

    sentiment_cols = [
        "source",
        "sent_mean",
        "sent_median",
        "sent_pos_share",
        "sent_neg_share",
    ]
    pd.DataFrame([row.__dict__ for row in summary_rows])[sentiment_cols].to_csv(
        output_dir / "sentiment.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Done. Sources: {len(source_labels)}")
    print(f"Results saved to: {output_dir}")
    if not morph.enabled:
        print(
            "Warning: pymorphy3 not installed. Lemmatization/POS fallback to token-based mode."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparative analysis of news feeds: tokenization, stopwords, lemmatization, stemming, TF-IDF, POS, sentiment."
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=Path("articles"),
        help="Directory with *_news.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "comparative_nlp",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N terms/lemmas/stems for output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        articles_dir=args.articles_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
