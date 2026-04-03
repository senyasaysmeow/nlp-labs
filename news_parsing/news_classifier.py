import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

script_dir = Path(__file__).parent
data_path = script_dir / "articles" / "ukr_pravda_news.json"

stopwords_ua = pd.read_csv(
    script_dir / "stopwords/stopwords_ua.txt", header=None, names=["w"]
)
stopwords_eng = pd.read_csv(
    script_dir / "stopwords/stopwords_eng.txt", header=None, names=["w"]
)
stop_words = set(stopwords_ua["w"].tolist() + stopwords_eng["w"].tolist())

CATEGORY_MAPPING = {
    "politics": "political",
    "economics": "economic",
    "life": "social",
    "sport": "social",
    "technologies": "technologies",
    "defence": "technologies",
    "other": "social",  # merged with social due to small sample
}

CATEGORY_NAMES_UA = {
    "political": "Політична",
    "economic": "Економічна",
    "social": "Соціальна",
    "technologies": "Технології/Оборона",
    "other": "Інше",
}


def load_news_data(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text: str) -> str:
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and numbers, keep Ukrainian and English letters
    text = re.sub(r"[^а-яіїєґa-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)


def prepare_data(news_data: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    texts = []
    original_categories = []
    mapped_categories = []

    for article in news_data:
        title = article.get("title", "")
        content = article.get("content", "")
        full_text = f"{title} {content}"

        category = article.get("category", "other")
        mapped_category = CATEGORY_MAPPING.get(category, "other")

        processed_text = preprocess_text(full_text)

        if processed_text and len(processed_text) > 10:
            texts.append(processed_text)
            original_categories.append(category)
            mapped_categories.append(mapped_category)

    return texts, original_categories, mapped_categories


def create_tfidf_features(texts: List[str], max_features: int = 5000) -> Tuple:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test) -> Dict:
    classifiers = {
        "Naive Bayes (MultinomialNB)": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0, max_iter=10000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ НАВЧАННЯ ТА ОЦІНКИ")
    print("=" * 70)

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")

        # Train
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        results[name] = {
            "model": clf,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "y_pred": y_pred,
        }

        print(f"Точність: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"F1-міра (macro): {f1_macro:.4f}")
        print(f"F1-міра (weighted): {f1_weighted:.4f}")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_name = name

    print("\n" + "=" * 70)
    print(f"НАЙКРАЩА МОДЕЛЬ: {best_name}")
    print(f"Найкраща точність: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print("=" * 70)

    # Перевірка чи досягнуто цільову точність (>60%)
    if best_accuracy >= 0.6:
        print("\n[УСПІХ] Цільову точність (>60%) досягнуто!")
    else:
        print(
            f"\n[УВАГА] Цільову точність (>60%) не досягнуто. Поточна: {best_accuracy * 100:.2f}%"
        )

    results["best"] = {
        "name": best_name,
        "model": best_model,
        "accuracy": best_accuracy,
    }

    return results


def cross_validate_best_model(X, y, best_clf_name: str) -> float:
    clf_map = {
        "Naive Bayes (MultinomialNB)": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0, max_iter=10000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    clf = clf_map.get(best_clf_name)
    if clf is None:
        return 0.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print(f"\nОцінки крос-валідації (5-fold): {cv_scores}")
    print(
        f"Середня точність CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )

    return cv_scores.mean()


def print_detailed_report(y_test, y_pred, label_encoder: LabelEncoder):
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНИЙ ЗВІТ КЛАСИФІКАЦІЇ")
    print("=" * 70)

    class_names = label_encoder.classes_
    class_names_ua = [CATEGORY_NAMES_UA.get(c, c) for c in class_names]

    report = classification_report(
        y_test, y_pred, target_names=class_names_ua, digits=4
    )
    print(report)


def plot_confusion_matrix(
    y_test, y_pred, label_encoder: LabelEncoder, save_path: str = None
):
    class_names = label_encoder.classes_
    class_names_ua = [CATEGORY_NAMES_UA.get(c, c) for c in class_names]

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_ua,
        yticklabels=class_names_ua,
    )
    plt.title("Матриця помилок")
    plt.xlabel("Передбачено")
    plt.ylabel("Справжнє")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nМатрицю помилок збережено у: {save_path}")

    plt.close()


def plot_model_comparison(results: Dict, save_path: str = None):
    model_names = []
    accuracies = []
    f1_scores_list = []

    for name, data in results.items():
        if name != "best":
            model_names.append(name.replace(" ", "\n"))
            accuracies.append(data["accuracy"])
            f1_scores_list.append(data["f1_macro"])

    x = np.arange(len(model_names))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2, accuracies, width, label="Accuracy", color="steelblue"
    )
    bars2 = ax.bar(
        x + width / 2, f1_scores_list, width, label="F1-score (macro)", color="coral"
    )

    # Add value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Додаємо лінію порогу 60%
    ax.axhline(y=0.6, color="red", linestyle="--", linewidth=2, label="Ціль (60%)")

    ax.set_xlabel("Модель")
    ax.set_ylabel("Оцінка")
    ax.set_title("Порівняння моделей")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.0)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Графік порівняння моделей збережено у: {save_path}")

    plt.close()


def main():
    print("\n" + "=" * 70)
    print("КЛАСИФІКАЦІЯ НОВИН З ВИКОРИСТАННЯМ НАВЧАННЯ З УЧИТЕЛЕМ")
    print(
        "Класифікація новин за сферами (економічна, політична, соціальна та технології)"
    )
    print("=" * 70)

    # 1. Завантаження даних
    print(f"\nЗавантаження даних з: {data_path}")
    news_data = load_news_data(data_path)
    print(f"    Всього новинних статей: {len(news_data)}")

    # 2. Підготовка даних
    print("\nПідготовка та попередня обробка даних...")
    texts, original_categories, mapped_categories = prepare_data(news_data)
    print(f"    Оброблено статей: {len(texts)}")

    # Вивід розподілу категорій
    from collections import Counter

    print("\n    Розподіл оригінальних категорій:")
    for cat, count in sorted(Counter(original_categories).items()):
        print(f"      {cat}: {count}")

    print("\n    Розподіл згрупованих категорій:")
    for cat, count in sorted(Counter(mapped_categories).items()):
        print(f"      {CATEGORY_NAMES_UA.get(cat, cat)}: {count}")

    # 3. Створення TF-IDF ознак
    print("\nСтворення TF-IDF ознак...")
    X, vectorizer = create_tfidf_features(texts)
    print(f"    Розмір матриці ознак: {X.shape}")
    print(f"    Кількість ознак (слова/n-грами): {X.shape[1]}")

    # 4. Кодування міток
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(mapped_categories)
    print("\nКодування міток завершено")
    print(f"    Класи: {list(label_encoder.classes_)}")

    # 5. Розділення даних
    print("\nРозділення даних (80% навчання, 20% тест)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Навчальних зразків: {X_train.shape[0]}")
    print(f"    Тестових зразків: {X_test.shape[0]}")

    # 6. Навчання та оцінка класифікаторів
    print("\nНавчання класифікаторів...")
    results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)

    # 7. Крос-валідація для найкращої моделі
    print("\nКрос-валідація для найкращої моделі...")
    best_name = results["best"]["name"]
    cv_accuracy = cross_validate_best_model(X, y, best_name)

    # 8. Detailed report for best model
    best_y_pred = results[best_name]["y_pred"]
    print_detailed_report(y_test, best_y_pred, label_encoder)

    # 9. Побудова матриці помилок
    print("\nГенерація візуалізацій...")
    plot_confusion_matrix(
        y_test, best_y_pred, label_encoder, script_dir / "confusion_matrix.png"
    )

    # Побудова порівняння моделей
    plot_model_comparison(results, script_dir / "model_comparison.png")

    # 10. Demonstrate predictions
    best_model = results["best"]["model"]
    test_idx = [i for i in range(len(texts))][-X_test.shape[0] :]
    sample_texts_subset = [texts[i] for i in test_idx[:10]]
    sample_categories_subset = [mapped_categories[i] for i in test_idx[:10]]

    return results


if __name__ == "__main__":
    main()
