import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Завантаження мовних моделей
nlp_uk = spacy.load("uk_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# =============================================================================
# ЗАВАНТАЖЕННЯ КАРКАСІВ ТЕМАТИК З JSON ФАЙЛІВ
# =============================================================================

# Шлях до директорії з JSON файлами
KEYWORDS_DIR = os.path.dirname(os.path.abspath(__file__))

# Список JSON файлів з каркасами
KEYWORDS_FILES = [
    "keywords_radioelectronics.json",
    "keywords_programming.json",
    "keywords_mechanical.json",
]


def load_keywords_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_topics():
    topic_keywords = {}
    topic_names = {}

    for filename in KEYWORDS_FILES:
        filepath = os.path.join(KEYWORDS_DIR, filename)
        if os.path.exists(filepath):
            data = load_keywords_from_json(filepath)
            topic_id = data["topic_id"]
            topic_keywords[topic_id] = data["keywords"]
            topic_names[topic_id] = data["topic_name"]

    # Додаємо "невідому тематику"
    topic_names["unknown"] = {"uk": "Невідома тематика", "en": "Unknown Topic"}

    return topic_keywords, topic_names


# Глобальне завантаження каркасів при імпорті модуля
TOPIC_KEYWORDS, TOPIC_NAMES = load_all_topics()


# =============================================================================
# ФУНКЦІЇ ОБРОБКИ ТЕКСТУ
# =============================================================================


def detect_language(text):
    # Простий метод: перевірка наявності кириличних символів
    cyrillic_count = len(re.findall(r"[а-яА-ЯіІїЇєЄґҐ]", text))
    latin_count = len(re.findall(r"[a-zA-Z]", text))

    if cyrillic_count > latin_count:
        return "uk"
    else:
        return "en"


def preprocess_text(text, lang="uk"):
    # Вибір моделі NLP
    nlp = nlp_uk if lang == "uk" else nlp_en

    # Видалення зайвих символів
    text = text.lower()
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Обробка через spacy
    doc = nlp(text)

    # Лематизація з фільтрацією стоп-слів та POS-тегуванням
    # Залишаємо лише іменники, дієслова, прикметники
    relevant_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
    tokens = []

    for token in doc:
        if not token.is_stop and token.pos_ in relevant_pos and len(token.lemma_) > 2:
            tokens.append(token.lemma_.lower())

    return tokens


def create_topic_vectors(lang="uk"):
    # Об'єднання ключових слів у документи
    topic_documents = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        topic_documents[topic] = " ".join(keywords[lang])

    return topic_documents


def classify_text_tfidf(text, threshold=0.05):
    # Визначення мови
    lang = detect_language(text)

    # Попередня обробка
    tokens = preprocess_text(text, lang)
    processed_text = " ".join(tokens)

    # Створення документів для тематик
    topic_docs = create_topic_vectors(lang)

    # Формування корпусу: [вхідний текст, тема1, тема2, тема3]
    documents = [processed_text] + list(topic_docs.values())
    topic_names = list(topic_docs.keys())

    # TF-IDF векторизація
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Обчислення косинусної подібності
    input_vector = tfidf_matrix[0:1]
    topic_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(input_vector, topic_vectors)[0]

    # Формування словника з результатами
    scores = {topic_names[i]: similarities[i] for i in range(len(topic_names))}

    # Визначення найкращої тематики
    max_score = max(similarities)

    if max_score < threshold:
        return "unknown", scores

    best_topic_idx = np.argmax(similarities)
    best_topic = topic_names[best_topic_idx]

    return best_topic, scores


def get_pos_tags(text, lang="uk"):
    nlp = nlp_uk if lang == "uk" else nlp_en

    doc = nlp(text)
    return [
        (token.text, token.pos_, token.lemma_) for token in doc if not token.is_space
    ]


def classify_and_display(text, threshold=0.05):
    print("\n" + "=" * 70)
    print("АНАЛІЗ ТЕКСТУ")
    print("=" * 70)

    # Визначення мови
    lang = detect_language(text)
    lang_name = "Українська" if lang == "uk" else "English"

    print(f'\nВхідний текст: "{text[:100]}{"..." if len(text) > 100 else ""}"')
    print(f"Визначена мова: {lang_name}")

    # POS-тегування
    pos_tags = get_pos_tags(text, lang)
    if pos_tags:
        print("\nPOS-теги (перші 10):")
        for word, pos, lemma in pos_tags[:10]:
            print(f"  {word:20} -> {pos:8} (лема: {lemma})")

    # Класифікація
    topic, scores = classify_text_tfidf(text, threshold)

    # Вивід результатів
    print(f"\n{'─' * 50}")
    print("РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ (TF-IDF + Косинусна подібність):")
    print(f"{'─' * 50}")

    for topic_key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        topic_name = TOPIC_NAMES[topic_key][lang]
        bar = "█" * int(score * 30)
        print(f"  {topic_name:25} : {score:.4f} {bar}")

    print(f"\n{'═' * 50}")
    result_name = TOPIC_NAMES[topic][lang]
    print(f"  ВИЗНАЧЕНА ТЕМАТИКА: {result_name}")
    print(f"{'═' * 50}")

    return topic, lang, scores


# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================


def main():
    print("\n" + "═" * 70)
    print("   КЛАСИФІКАТОР ТЕКСТУ ЗА ТЕМАТИКАМИ")
    print("   (Радіоелектроніка / Програмування / Машинобудування)")
    print("═" * 70)

    # Приклади тестових текстів
    test_texts = {
        "uk": [
            "Для збирання підсилювача потрібні транзистори, резистори та конденсатори. "
            "Схема живиться від напруги 12 вольт. Сигнал подається на вхід через антену.",
            "Python є інтерпретованою мовою програмування. Функції визначаються ключовим словом def. "
            "Для роботи з масивами використовуються цикли for та while.",
            "Двигун внутрішнього згоряння має поршні, які рухаються в циліндрах. "
            "Колінчастий вал передає обертання через редуктор на ведучі колеса.",
            "Сьогодні гарна погода, птахи співають, а квіти цвітуть у саду.",
        ],
        "en": [
            "The amplifier circuit uses transistors, resistors, and capacitors. "
            "The signal is modulated at high frequency and transmitted through the antenna.",
            "Python is an interpreted programming language. Functions are defined using the def keyword. "
            "Arrays can be processed using for and while loops with conditional statements.",
            "The internal combustion engine has pistons moving inside cylinders. "
            "The crankshaft transfers rotation through the gearbox to the driving wheels.",
            "The weather is beautiful today, birds are singing in the garden.",
        ],
    }

    print("\nОберіть режим роботи:")
    print("1 - Тестові приклади (українською)")
    print("2 - Тестові приклади (англійською)")
    print("3 - Введення власного тексту")
    print("0 - Вихід")

    mode = int(input("\nВаш вибір: "))

    if mode == 1:
        print("\n" + "─" * 70)
        print("ТЕСТУВАННЯ НА УКРАЇНСЬКИХ ТЕКСТАХ")
        print("─" * 70)

        for i, text in enumerate(test_texts["uk"], 1):
            print(f"\n>>> Тест {i}")
            classify_and_display(text)

    elif mode == 2:
        print("\n" + "─" * 70)
        print("TESTING ON ENGLISH TEXTS")
        print("─" * 70)

        for i, text in enumerate(test_texts["en"], 1):
            print(f"\n>>> Test {i}")
            classify_and_display(text)

    elif mode == 3:
        print("\nВведіть текст для класифікації (Enter для завершення):")
        while True:
            text = input("\n>>> Текст: ").strip()
            if not text:
                break
            classify_and_display(text)

    print("\n" + "═" * 70)
    print("Програму завершено")
    print("═" * 70)


if __name__ == "__main__":
    main()
