import json
import pandas as pd
import re
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wordcloud import WordCloud
import os
from pathlib import Path

script_dir = Path(__file__).parent
dir_path = script_dir / "analysis/"
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


def get_time_period(dt):
    hour = dt.hour
    if 0 <= hour < 12:
        return "Ранок"
    elif 12 <= hour < 18:
        return "Обід"
    else:
        return "Вечір"


def load_data():
    all_news = []
    for file in ["articles/ukr_pravda_news.json", "articles/suspilne_news.json"]:
        with open(file, "r", encoding="utf-8") as f:
            news = json.load(f)
            all_news.extend(news)

    processed = []
    for item in all_news:
        time_str = item["time"]
        dt = datetime.fromisoformat(time_str)
        period = get_time_period(dt)
        day_str = dt.strftime("%d.%m.%Y")

        text = item.get("title") + " " + item.get("content")
        words = clean_text(text)

        processed.append(
            {
                "date_obj": dt.date(),
                "day_str": day_str,
                "period": period,
                "words": words,
            }
        )

    return processed


def find_top3_from_table(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    terms = df["Топ 5"].dropna()

    freq = Counter(terms.str.strip().str.lower())
    top3 = [w for w, _ in freq.most_common(3)]
    print("Топ-3 терміни за весь період моніторингу:")
    for rank, (w, cnt) in enumerate(freq.most_common(3), 1):
        print(f"  {rank}. «{w}» — зустрічався у Топ-5 {cnt} разів")
    return top3


def build_daily_timeseries(articles, top3):
    dates = sorted({a["date_obj"] for a in articles})

    rows = []
    for d in dates:
        day_articles = [a for a in articles if a["date_obj"] == d]
        all_words = [w for a in day_articles for w in a["words"]]
        word_freq = Counter(all_words)

        row = {"Дата": d.strftime("%d.%m.%Y")}
        for term in top3:
            row[term] = word_freq.get(term, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_timeseries(df, top3):
    COLORS = ["#e63946", "#2a9d8f", "#f4a261"]
    MARKERS = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(13, 6))

    x = range(len(df))
    for i, term in enumerate(top3):
        ax.plot(
            x,
            df[term],
            marker=MARKERS[i],
            color=COLORS[i],
            linewidth=2,
            markersize=6,
            label=f"«{term}»",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Дата"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_title(
        "Часові ряди топ-3 термінів (сумарна частота за день)", fontsize=14, pad=14
    )
    ax.set_xlabel("Дата", fontsize=11)
    ax.set_ylabel("Кількість згадувань за день", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{dir_path}top3_linechart.png", dpi=150)
    print(f"Графік збережено у {dir_path}top3_linechart.png")


def main():
    data = load_data()
    if not data:
        print("No data available.")
        return

    # Group by Day and Period
    grouped = {}
    for item in data:
        key = (item["day_str"], item["period"], item["date_obj"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].extend(item["words"])

    # Sort groups by date and then period order
    period_order = {"Ранок": 0, "Обід": 1, "Вечір": 2}
    sorted_keys = sorted(grouped.keys(), key=lambda x: (x[2], period_order[x[1]]))

    # Prepare Table 1
    table_rows = []
    plot_data = []
    all_top_words = Counter()

    # Group by Date first
    dates = sorted(list(set(k[0] for k in sorted_keys)))

    for day_idx, current_day_str in enumerate(dates, 1):
        # Find periods for this day
        day_keys = [k for k in sorted_keys if k[0] == current_day_str]

        for period_idx, key in enumerate(day_keys):
            day_str, period, date_obj = key
            words = grouped[key]

            counter = Counter(words)
            top5 = counter.most_common(5)

            sum_freq = sum(freq for word, freq in top5)

            for i in range(5):
                word, freq = top5[i]

                # Populate day_val only on the first row of the entire day
                if period_idx == 0 and i == 0:
                    day_val = f"День {day_idx}\n{current_day_str}"
                else:
                    day_val = ""

                # Populate period_val and sum_val only on the first row of the period
                if i == 0:
                    time_val = f"{period}"
                    sum_val = sum_freq
                else:
                    time_val = ""
                    sum_val = ""

                table_rows.append(
                    {
                        "День": day_val,
                        "Час": time_val,
                        "Топ 5": word,
                        "Частота": freq if freq > 0 else "",
                        "Сума частот": sum_val,
                    }
                )

                if word:
                    all_top_words[word] += freq

            plot_data.append({"datetime": f"{day_str} {period}", "sum_freq": sum_freq})

    table = pd.DataFrame(table_rows)
    table.to_csv(f"{dir_path}table.csv", index=False, encoding="utf-8-sig")
    print(f"Таблиця збережена у {dir_path}table.csv")

    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate_from_frequencies(all_top_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Хмара слів (Топ 5 термінів)")
    plt.tight_layout()
    plt.savefig(f"{dir_path}wordcloud.png")
    print(f"Хмара слів збережена у {dir_path}wordcloud.png")

    top3 = find_top3_from_table(f"{dir_path}table.csv")
    articles = load_data()
    top3_timeseries = build_daily_timeseries(articles, top3)
    top3_timeseries.to_csv(
        f"{dir_path}top3_timeseries.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nЧасові ряди збережено у {dir_path}top3_timeseries.csv")
    print(top3_timeseries.to_string(index=False))
    plot_timeseries(top3_timeseries, top3)


if __name__ == "__main__":
    main()
