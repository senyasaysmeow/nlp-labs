import json
import re
import os
import math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

script_dir = Path(__file__).parent
dir_path = script_dir / "text_analysis/"
os.makedirs(dir_path, exist_ok=True)

stopwords_ua = pd.read_csv(
    script_dir / "stopwords/stopwords_ua.txt", header=None, names=["w"]
)
stopwords_eng = pd.read_csv(
    script_dir / "stopwords/stopwords_eng.txt", header=None, names=["w"]
)
stop_words = set(stopwords_ua["w"].tolist() + stopwords_eng["w"].tolist())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яяіїєґa-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words


def load_news(filepath="articles/ukr_pravda_news.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        news = json.load(f)

    documents = []
    for item in news:
        text = f"{item.get('title', '')} {item.get('content', '')}"
        words = clean_text(text)
        documents.append(
            {
                "title": item.get("title", ""),
                "category": item.get("category", "unknown"),
                "words": words,
                "raw_text": text,
            }
        )
    return documents


# =============================================================================
# TF-IDF АНАЛІЗ
# =============================================================================


def compute_tf(words):
    word_count = Counter(words)
    total_words = len(words)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf


def compute_idf(documents):
    n_docs = len(documents)
    word_doc_count = defaultdict(int)

    for doc in documents:
        unique_words = set(doc["words"])
        for word in unique_words:
            word_doc_count[word] += 1

    idf = {}
    for word, doc_count in word_doc_count.items():
        idf[word] = math.log(n_docs / doc_count) + 1  # +1 для згладжування

    return idf


def compute_tfidf(documents):
    idf = compute_idf(documents)

    tfidf_results = []
    for doc in documents:
        tf = compute_tf(doc["words"])
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_results.append(
            {
                "title": doc["title"][:50] + "..."
                if len(doc["title"]) > 50
                else doc["title"],
                "tfidf": tfidf,
            }
        )

    return tfidf_results


def get_corpus_tfidf_scores(tfidf_results):
    aggregated = defaultdict(float)
    for result in tfidf_results:
        for word, score in result["tfidf"].items():
            aggregated[word] += score
    return dict(aggregated)


def analyze_tfidf(documents):
    print("\n" + "=" * 60)
    print("TF-IDF АНАЛІЗ")
    print("=" * 60)

    tfidf_results = compute_tfidf(documents)
    corpus_tfidf = get_corpus_tfidf_scores(tfidf_results)

    # Топ-25 слів за TF-IDF
    top_words = sorted(corpus_tfidf.items(), key=lambda x: x[1], reverse=True)[:25]

    print("\nТоп-25 слів за сумарним TF-IDF: \n")
    for i, (word, score) in enumerate(top_words, 1):
        print(f"{i:2}. {word:13} {score:.4f}")

    # Збереження результатів
    tfidf_df = pd.DataFrame(top_words, columns=["Слово", "TF-IDF"])
    tfidf_df.to_csv(f"{dir_path}/tfidf_scores.csv", index=False, encoding="utf-8-sig")

    # Візуалізація
    _, ax = plt.subplots(figsize=(12, 8))
    words = [w for w, _ in top_words]
    scores = [s for _, s in top_words]

    ax.barh(range(len(words)), scores, color="steelblue")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("TF-IDF Score")
    ax.set_title("Топ-25 слів за TF-IDF")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{dir_path}/tfidf_chart.png", dpi=150)
    plt.close()

    print(f"\nГрафік збережено: {dir_path}/tfidf_chart.png")

    return corpus_tfidf


# =============================================================================
# ЛЕКСИЧНА ДИСПЕРСІЯ
# =============================================================================


def compute_lexical_dispersion(documents, target_words):
    print("\n" + "=" * 60)
    print("ЛЕКСИЧНА ДИСПЕРСІЯ")
    print("=" * 60)

    # Збираємо всі слова з позиціями
    all_words = []
    doc_boundaries = []

    for doc in documents:
        all_words.extend(doc["words"])
        doc_boundaries.append(len(all_words))

    # Знаходимо позиції кожного цільового слова
    word_positions = defaultdict(list)
    for i, word in enumerate(all_words):
        if word in target_words:
            word_positions[word].append(i)

    # Обчислення метрик дисперсії
    dispersion_stats = []
    total_words = len(all_words)

    print(f"\nЗагальна кількість слів у корпусі: {total_words}")
    print(f"\nАналіз дисперсії для {len(target_words)} ключових слів: \n")

    for word in target_words:
        positions = word_positions[word]
        freq = len(positions)

        if freq > 1:
            # Середня відстань між входженнями
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            # Коефіцієнт варіації (нормалізована дисперсія)
            cv = std_gap / avg_gap if avg_gap > 0 else 0
            # Рівномірність розподілу (0-1, де 1 = рівномірний)
            uniformity = 1 - (cv / 2) if cv < 2 else 0
        else:
            avg_gap = 0
            std_gap = 0
            uniformity = 0

        dispersion_stats.append(
            {
                "word": word,
                "frequency": freq,
                "avg_gap": avg_gap,
                "std_gap": std_gap,
                "uniformity": uniformity,
            }
        )

        print(
            f"{word:13} | Частота: {freq:4} | Середній інтервал: {avg_gap:8.1f} | "
            f"Рівномірність: {uniformity:.3f}"
        )

    # Візуалізація
    _, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(target_words)))

    for idx, word in enumerate(target_words):
        positions = word_positions[word]
        y_values = [idx] * len(positions)
        ax.scatter(positions, y_values, c=[colors[idx]], label=word, s=3, alpha=0.7)

    # Позначаємо межі документів
    for boundary in doc_boundaries[:-1]:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.3)

    ax.set_yticks(range(len(target_words)))
    ax.set_yticklabels(target_words)
    ax.set_xlabel("Позиція в корпусі (порядок слів)")
    ax.set_ylabel("Ключові слова")
    ax.set_title("Лексична дисперсія: розподіл ключових слів по корпусу")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{dir_path}/lexical_dispersion.png", dpi=150)
    plt.close()

    # Збереження статистики
    disp_df = pd.DataFrame(dispersion_stats)
    disp_df.to_csv(
        f"{dir_path}/dispersion_stats.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\nГрафік дисперсії збережено: {dir_path}/lexical_dispersion.png")

    return dispersion_stats, word_positions


# =============================================================================
# РОЗПОДІЛ ДОВЖИНИ СЛІВ
# =============================================================================


def analyze_word_length_distribution(documents):
    print("\n" + "=" * 60)
    print("РОЗПОДІЛ ДОВЖИНИ СЛІВ")
    print("=" * 60)

    # Збираємо всі слова та їх довжини
    all_words = []
    for doc in documents:
        all_words.extend(doc["words"])

    word_lengths = [len(word) for word in all_words]

    # Статистика
    length_freq = Counter(word_lengths)
    total_words = len(word_lengths)

    print(f"\nЗагальна кількість слів: {total_words}")
    print(f"Середня довжина слова: {np.mean(word_lengths):.2f}")
    print(f"Медіанна довжина слова: {np.median(word_lengths):.0f}")
    print(f"Стандартне відхилення: {np.std(word_lengths):.2f}")
    print(f"Мінімальна довжина: {min(word_lengths)}")
    print(f"Максимальна довжина: {max(word_lengths)}")

    # Розподіл за довжиною
    distribution_data = []
    for length in sorted(length_freq.keys()):
        count = length_freq[length]
        prob = count / total_words
        distribution_data.append(
            {"length": length, "count": count, "probability": prob}
        )

    # Візуалізація
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Гістограма частот
    ax1 = axes[0]
    lengths_list = sorted(length_freq.keys())
    counts = [length_freq[l] for l in lengths_list]
    ax1.bar(lengths_list, counts, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Довжина слова")
    ax1.set_ylabel("Кількість слів")
    ax1.set_title("Гістограма довжини слів")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Ймовірнісний розподіл
    ax2 = axes[1]
    probs = [length_freq[l] / total_words for l in lengths_list]
    ax2.bar(lengths_list, probs, color="coral", edgecolor="black")
    ax2.set_xlabel("Довжина слова")
    ax2.set_ylabel("Ймовірність P(X=x)")
    ax2.set_title("Ймовірнісний розподіл")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Кумулятивний розподіл
    ax3 = axes[2]
    cumulative_probs = np.cumsum(probs)
    ax3.step(lengths_list, cumulative_probs, where="mid", color="green", linewidth=2)
    ax3.fill_between(
        lengths_list, cumulative_probs, step="mid", alpha=0.3, color="green"
    )
    ax3.set_xlabel("Довжина слова")
    ax3.set_ylabel("P(X <= x)")
    ax3.set_title("Кумулятивна функція розподілу")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{dir_path}/word_length_distribution.png", dpi=150)
    plt.close()

    # Збереження даних
    dist_df = pd.DataFrame(distribution_data)
    dist_df.to_csv(
        f"{dir_path}/word_length_distribution.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\nГрафіки збережено: {dir_path}/word_length_distribution.png")

    return distribution_data


# =============================================================================
# БІГРАМНИЙ АНАЛІЗ
# =============================================================================


def analyze_bigrams(documents):
    print("\n" + "=" * 60)
    print("БІГРАМНИЙ АНАЛІЗ")
    print("=" * 60)

    # Збираємо всі слова
    all_words = [word for doc in documents for word in doc["words"]]

    # Створюємо BigramCollocationFinder
    finder = BigramCollocationFinder.from_words(all_words)
    finder.apply_freq_filter(3)  # Фільтруємо рідкісні біграми

    bigram_measures = BigramAssocMeasures()
    word_freq = Counter(all_words)
    total_words = len(all_words)
    total_bigrams = total_words - 1

    print(f"\nЗагальна кількість слів: {total_words}")
    print(f"Загальна кількість біграм: {total_bigrams}")
    print(f"Унікальних біграм (freq>=3): {len(list(finder.ngram_fd.items()))}")

    # Топ-30 найчастіших біграм
    top_bigrams = finder.ngram_fd.most_common(30)

    print("\nТоп-30 найчастіших біграм:")
    print("-" * 60)

    bigram_data = []
    for i, ((w1, w2), count) in enumerate(top_bigrams, 1):
        joint_prob = count / total_bigrams
        cond_prob = count / word_freq[w1]
        pmi = finder.score_ngram(bigram_measures.pmi, w1, w2)

        bigram_data.append(
            {
                "bigram": f"{w1} {w2}",
                "count": count,
                "joint_prob": joint_prob,
                "cond_prob": cond_prob,
                "pmi": pmi,
            }
        )

        print(
            f"{i:2}. '{w1} {w2}' : {count:4} | P(w1,w2)={joint_prob:.5f} | "
            f"P(w2|w1)={cond_prob:.3f} | PMI={pmi:.2f}"
        )

    # Топ-25 за PMI
    top_pmi_raw = finder.nbest(bigram_measures.pmi, 25)
    top_pmi = [
        {
            "bigram": f"{w1} {w2}",
            "count": finder.ngram_fd[(w1, w2)],
            "pmi": finder.score_ngram(bigram_measures.pmi, w1, w2),
        }
        for w1, w2 in top_pmi_raw
    ]

    print("\nТоп-25 біграм за PMI:")
    print("-" * 50)
    for i, item in enumerate(top_pmi, 1):
        print(
            f"{i:2}. '{item['bigram']}' : частота={item['count']:3} | PMI={item['pmi']:.3f}"
        )

    # Візуалізація
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Топ-15 біграм за частотою
    ax1 = axes[0]
    top15 = bigram_data[:15]
    bigram_labels = [b["bigram"] for b in top15]
    counts = [b["count"] for b in top15]

    ax1.barh(range(len(bigram_labels)), counts, color="teal")
    ax1.set_yticks(range(len(bigram_labels)))
    ax1.set_yticklabels(bigram_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Частота")
    ax1.set_title("Топ-15 біграм за частотою")
    ax1.grid(axis="x", alpha=0.3)

    # 2. Топ-15 біграм за PMI
    ax2 = axes[1]
    top15_pmi = top_pmi[:15]
    pmi_labels = [b["bigram"] for b in top15_pmi]
    pmi_values = [b["pmi"] for b in top15_pmi]

    ax2.barh(range(len(pmi_labels)), pmi_values, color="purple")
    ax2.set_yticks(range(len(pmi_labels)))
    ax2.set_yticklabels(pmi_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("PMI (Pointwise Mutual Information)")
    ax2.set_title("Топ-15 біграм за PMI")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{dir_path}/bigram_analysis.png", dpi=150)
    plt.close()

    # Збереження даних
    bigram_df = pd.DataFrame(bigram_data)
    bigram_df.to_csv(
        f"{dir_path}/bigram_frequency.csv", index=False, encoding="utf-8-sig"
    )

    pmi_df = pd.DataFrame(top_pmi)
    pmi_df.to_csv(f"{dir_path}/bigram_pmi.csv", index=False, encoding="utf-8-sig")

    print(f"\nГрафіки збережено: {dir_path}/bigram_analysis.png")

    return bigram_data, top_pmi


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================


def main():
    print("=" * 60)
    print("ЧАСТОТНИЙ ТА ЙМОВІРНІСНИЙ АНАЛІЗ НОВИН")
    print("=" * 60)

    # Завантаження даних
    print("\nЗавантаження новин...")
    documents = load_news("articles/ukr_pravda_news.json")
    print(f"Завантажено {len(documents)} документів")

    # Статистика корпусу
    total_words = sum(len(doc["words"]) for doc in documents)
    unique_words = len(set(word for doc in documents for word in doc["words"]))
    print(f"Загальна кількість слів: {total_words}")
    print(f"Унікальних слів: {unique_words}")
    print(f"Лексична різноманітність (TTR): {unique_words / total_words:.4f}")

    # 1. TF-IDF аналіз
    tfidf_scores = analyze_tfidf(documents)

    # 2. Лексична дисперсія
    # Вибираємо топ-10 ключових слів для аналізу дисперсії
    top_tfidf_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    target_words = [w for w, _ in top_tfidf_words]
    dispersion_stats, _ = compute_lexical_dispersion(documents, target_words)

    # 3. Розподіл довжини слів
    word_length_dist = analyze_word_length_distribution(documents)

    # 4. Біграмний аналіз
    bigram_data, pmi_data = analyze_bigrams(documents)

    # Підсумок
    print("\n" + "=" * 60)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"\nРезультати збережено в папці: {dir_path}/")
    print("Файли:")
    print("  - tfidf_scores.csv - TF-IDF оцінки")
    print("  - tfidf_chart.png - Візуалізація TF-IDF")
    print("  - dispersion_stats.csv - Статистика дисперсії")
    print("  - lexical_dispersion.png - Графік дисперсії")
    print("  - word_length_distribution.csv - Розподіл довжини слів")
    print("  - word_length_distribution.png - Графіки розподілу")
    print("  - bigram_frequency.csv - Частоти біграм")
    print("  - bigram_pmi.csv - PMI біграм")
    print("  - bigram_analysis.png - Візуалізація біграм")


if __name__ == "__main__":
    main()
