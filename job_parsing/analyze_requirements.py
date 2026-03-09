import json
import re
from collections import Counter

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# 0. Завантаження моделей та ресурсів
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS_NLTK = set(stopwords.words("english"))
STOP_WORDS_SPACY = nlp.Defaults.stop_words
STOP_WORDS = STOP_WORDS_NLTK | STOP_WORDS_SPACY

stemmer = PorterStemmer()
wnl = WordNetLemmatizer()

# Додаткові стоп-слова специфічні для вакансій (не несуть аналітичної цінності)
CUSTOM_STOP_WORDS = {
    "experience",
    "work",
    "team",
    "company",
    "role",
    "looking",
    "join",
    "ability",
    "strong",
    "working",
    "etc",
    "e.g",
    "eg",
    "year",
    "years",
    "also",
    "least",
    "including",
    "well",
    "new",
    "us",
    "one",
    "use",
    "using",
    "used",
    "day",
    "would",
    "need",
    "ensure",
    "make",
    "like",
    "good",
    "plus",
    "based",
    "across",
    "level",
    "high",
    "key",
    "must",
    "related",
    "knowledge",
    "understanding",
    "skill",
    "skills",
    "candidate",
    "ideal",
    "offer",
    "offers",
    "responsibilities",
    "requirement",
    "requirements",
    "required",
    "nice",
    "hiring",
    "description",
    "job",
    "position",
    "opportunity",
    "opportunities",
    "want",
    "help",
    "look",
    "seek",
    "seeking",
    "right",
    "create",
    "full",
    "best",
    "real",
    "world",
    "global",
    "part",
    "first",
    "within",
    "time",
    "support",
    "people",
    "professional",
    "environment",
    "take",
    "develop",
    "provide",
    "different",
    "major",
    "apply",
    "available",
    "following",
    "willingness",
}

# ---------------------------------------------------------------------------
# 1. Завантаження даних
# ---------------------------------------------------------------------------
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "jobs", "djinni_jobs.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    jobs = json.load(f)

print(f"Завантажено вакансій: {len(jobs)}")

# ---------------------------------------------------------------------------
# 2. Фільтрація — залишаємо лише вакансії, що відносяться до Data Analyst
# ---------------------------------------------------------------------------
ANALYST_KEYWORDS = {
    "data analyst",
    "product analyst",
    "bi analyst",
    "bi developer",
    "business intelligence",
    "analytics",
    "marketing analyst",
    "senior data analyst",
    "middle data analyst",
    "junior data analyst",
}


def is_data_analyst_job(job: dict) -> bool:
    title_lower = job["title"].lower()
    for kw in ANALYST_KEYWORDS:
        if kw in title_lower:
            return True
    # Додатково: якщо в контенті згадується "data analyst" у перших 500 символах
    content_start = job.get("content", "")[:500].lower()
    if "data analyst" in content_start:
        return True
    return False


filtered_jobs = [j for j in jobs if is_data_analyst_job(j)]
print(f"Після фільтрації (Data Analyst-related): {len(filtered_jobs)}")

# ---------------------------------------------------------------------------
# 3. Нормалізація тексту
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Нормалізація: нижній регістр, видалення URL, email, спецсимволів, emoji."""
    text = text.lower()
    # Видалення URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Видалення email
    text = re.sub(r"\S+@\S+", " ", text)
    # Видалення emoji та unicode символів
    text = re.sub(
        r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
        r"\U0001f900-\U0001f9ff\U00002702-\U000027b0\U0001fa00-\U0001fa6f"
        r"\U0001fa70-\U0001faff\U00002600-\U000026ff\U0000fe0f]+",
        " ",
        text,
    )
    # Зберегти деякі технічні терміни з символами (C++, C#, A/B)
    text = re.sub(r"a/b", "ab_test", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    # Видалення пунктуації (крім _ для збережених термінів)
    text = re.sub(r"[^\w\s]", " ", text)
    # Видалення зайвих пробілів
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 4. Об'єднання контенту вакансій
# ---------------------------------------------------------------------------
all_texts_raw = []
for job in filtered_jobs:
    combined = f"{job.get('title', '')} {job.get('content', '')}"
    all_texts_raw.append(combined)

corpus_raw = " ".join(all_texts_raw)
corpus_normalized = normalize_text(corpus_raw)

print(f"Довжина корпусу (символів): {len(corpus_normalized)}")

# ---------------------------------------------------------------------------
# 5. Токенізація (NLTK)
# ---------------------------------------------------------------------------
tokens_nltk = word_tokenize(corpus_normalized)
print(f"Кількість токенів (NLTK word_tokenize): {len(tokens_nltk)}")

# ---------------------------------------------------------------------------
# 6. Видалення стоп-слів
# ---------------------------------------------------------------------------


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Видалення стоп-слів (NLTK + SpaCy + custom)."""
    return [
        t
        for t in tokens
        if t not in STOP_WORDS
        and t not in CUSTOM_STOP_WORDS
        and len(t) > 2
        and not t.isdigit()
    ]


tokens_clean = remove_stop_words(tokens_nltk)
print(f"Токенів після видалення стоп-слів: {len(tokens_clean)}")

# ---------------------------------------------------------------------------
# 7. Лематизація (SpaCy)
# ---------------------------------------------------------------------------


def lemmatize_spacy(tokens: list[str]) -> list[str]:
    """Лематизація з використанням SpaCy."""
    text = " ".join(tokens)
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_space]


tokens_lemmatized_spacy = lemmatize_spacy(tokens_clean)
tokens_lemmatized_spacy = remove_stop_words(tokens_lemmatized_spacy)

# ---------------------------------------------------------------------------
# 8. Стемінг (NLTK Porter Stemmer)
# ---------------------------------------------------------------------------


def stem_tokens(tokens: list[str]) -> list[str]:
    """Стемінг з використанням NLTK PorterStemmer."""
    return [stemmer.stem(t) for t in tokens]


tokens_stemmed = stem_tokens(tokens_clean)

# ---------------------------------------------------------------------------
# 9. Лематизація (NLTK WordNet) — для порівняння
# ---------------------------------------------------------------------------


def lemmatize_nltk(tokens: list[str]) -> list[str]:
    """Лематизація з використанням NLTK WordNetLemmatizer."""
    return [wnl.lemmatize(t) for t in tokens]


tokens_lemmatized_nltk = lemmatize_nltk(tokens_clean)
tokens_lemmatized_nltk = remove_stop_words(tokens_lemmatized_nltk)

# ---------------------------------------------------------------------------
# 10. Аналіз n-gram (біграми для виявлення технологій та навичок)
# ---------------------------------------------------------------------------


def get_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Отримати n-грами зі списку токенів."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


bigrams = get_ngrams(tokens_lemmatized_spacy, 2)
trigrams = get_ngrams(tokens_lemmatized_spacy, 3)

# ---------------------------------------------------------------------------
# 11. Виявлення іменованих сутностей (SpaCy NER) — технології та навички
# ---------------------------------------------------------------------------


def extract_tech_entities(texts: list[str]) -> Counter:
    """Витяг технічних термінів через NER та шаблони."""
    tech_terms = []
    # Відомі технології та інструменти для пошуку
    TECH_PATTERNS = {
        "sql",
        "python",
        "tableau",
        "power bi",
        "powerbi",
        "excel",
        "bigquery",
        "big query",
        "clickhouse",
        "postgresql",
        "postgres",
        "mysql",
        "snowflake",
        "redshift",
        "looker",
        "looker studio",
        "google analytics",
        "ga4",
        "dax",
        "power query",
        "etl",
        "elt",
        "airflow",
        "dbt",
        "spark",
        "hadoop",
        "aws",
        "gcp",
        "azure",
        "docker",
        "git",
        "github",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy",
        "scikit",
        "tensorflow",
        "pytorch",
        "jupyter",
        "databricks",
        "metabase",
        "superset",
        "amplitude",
        "mixpanel",
        "hotjar",
        "firebase",
        "appsflyer",
        "gtm",
        "google tag manager",
        "jira",
        "confluence",
        "notion",
        "ab_test",
        "a/b test",
        "ab test",
        "machine learning",
        "deep learning",
        "statistics",
        "statistical",
        "r language",
        "r programming",
    }

    for text in texts:
        text_lower = text.lower()
        for term in TECH_PATTERNS:
            count = text_lower.count(term)
            tech_terms.extend([term] * count)

    return Counter(tech_terms)


tech_counter = extract_tech_entities(all_texts_raw)

# ---------------------------------------------------------------------------
# 12. Виведення результатів
# ---------------------------------------------------------------------------
SEPARATOR = "=" * 70

print(f"\n{SEPARATOR}")
print("АНАЛІЗ ДОМІНАНТНИХ ВИМОГ ДО ПОСАДИ АНАЛІТИКА ДАНИХ")
print(f"(на основі {len(filtered_jobs)} вакансій з Djinni)")
print(SEPARATOR)

# --- Етапи обробки ---
print("\n--- ЕТАПИ NLP-ОБРОБКИ ---")
print(f"  1. Фільтрація:        {len(jobs)} -> {len(filtered_jobs)} вакансій")
print(
    f"  2. Нормалізація:       {len(corpus_raw)} -> {len(corpus_normalized)} символів"
)
print(f"  3. Токенізація (NLTK): {len(tokens_nltk)} токенів")
print(f"  4. Стоп-слова:         {len(tokens_nltk)} -> {len(tokens_clean)} токенів")
print(
    f"  5. Лематизація SpaCy:  {len(tokens_clean)} -> {len(tokens_lemmatized_spacy)} лем"
)
print(f"  6. Стемінг NLTK:       {len(tokens_clean)} -> {len(tokens_stemmed)} стемів")
print(
    f"  7. Лематизація NLTK:   {len(tokens_clean)} -> {len(tokens_lemmatized_nltk)} лем"
)

# --- Топ слів після лематизації SpaCy ---
print(f"\n{SEPARATOR}")
print("ТОП-30 СЛІВ (Лематизація SpaCy)")
print(SEPARATOR)
freq_spacy = Counter(tokens_lemmatized_spacy)
for rank, (word, count) in enumerate(freq_spacy.most_common(30), 1):
    bar = "#" * (count // 5)
    print(f"  {rank:>2}. {word:<25} {count:>4}  {bar}")

# --- Топ слів після стемінгу NLTK ---
print(f"\n{SEPARATOR}")
print("ТОП-30 СЛІВ (Стемінг NLTK)")
print(SEPARATOR)
freq_stemmed = Counter(tokens_stemmed)
for rank, (word, count) in enumerate(freq_stemmed.most_common(30), 1):
    bar = "#" * (count // 5)
    print(f"  {rank:>2}. {word:<25} {count:>4}  {bar}")

# --- Топ слів після лематизації NLTK ---
print(f"\n{SEPARATOR}")
print("ТОП-30 СЛІВ (Лематизація NLTK WordNet)")
print(SEPARATOR)
freq_nltk_lem = Counter(tokens_lemmatized_nltk)
for rank, (word, count) in enumerate(freq_nltk_lem.most_common(30), 1):
    bar = "#" * (count // 5)
    print(f"  {rank:>2}. {word:<25} {count:>4}  {bar}")

# --- Технології та інструменти ---
print(f"\n{SEPARATOR}")
print("ТОП ТЕХНОЛОГІЙ ТА ІНСТРУМЕНТІВ (частота згадувань)")
print(SEPARATOR)
for rank, (tech, count) in enumerate(tech_counter.most_common(25), 1):
    bar = "#" * (count // 2)
    print(f"  {rank:>2}. {tech:<25} {count:>4}  {bar}")

# --- Біграми ---
print(f"\n{SEPARATOR}")
print("ТОП-25 БІГРАМ (пари слів)")
print(SEPARATOR)
freq_bigrams = Counter(bigrams)
for rank, (bg, count) in enumerate(freq_bigrams.most_common(25), 1):
    bar = "#" * (count // 2)
    print(f"  {rank:>2}. {bg:<35} {count:>4}  {bar}")

# --- Триграми ---
print(f"\n{SEPARATOR}")
print("ТОП-20 ТРИГРАМ (трійки слів)")
print(SEPARATOR)
freq_trigrams = Counter(trigrams)
for rank, (tg, count) in enumerate(freq_trigrams.most_common(20), 1):
    bar = "#" * (count // 2)
    print(f"  {rank:>2}. {tg:<45} {count:>4}  {bar}")

# ---------------------------------------------------------------------------
# 13. Категоризація вимог
# ---------------------------------------------------------------------------
print(f"\n{SEPARATOR}")
print("КАТЕГОРИЗАЦІЯ ДОМІНАНТНИХ ВИМОГ")
print(SEPARATOR)

CATEGORIES = {
    "Hard Skills (Технічні навички)": [
        "sql",
        "python",
        "excel",
        "statistics",
        "statistical",
        "machine learning",
        "ab_test",
        "etl",
        "dax",
        "power query",
        "modeling",
        "programming",
        "scripting",
        "automation",
    ],
    "BI & Візуалізація": [
        "tableau",
        "power bi",
        "powerbi",
        "looker",
        "looker studio",
        "metabase",
        "superset",
        "dashboard",
        "visualization",
        "reporting",
        "report",
    ],
    "Бази даних та Хмарні платформи": [
        "bigquery",
        "big query",
        "clickhouse",
        "postgresql",
        "postgres",
        "mysql",
        "snowflake",
        "redshift",
        "databricks",
        "aws",
        "gcp",
        "azure",
    ],
    "Аналітичні платформи": [
        "google analytics",
        "ga4",
        "amplitude",
        "mixpanel",
        "hotjar",
        "firebase",
        "appsflyer",
        "gtm",
    ],
    "Soft Skills": [
        "communication",
        "english",
        "teamwork",
        "presentation",
        "problem solving",
        "analytical thinking",
        "attention detail",
        "proactive",
        "independent",
    ],
}


def count_category(texts: list[str], keywords: list[str]) -> dict[str, int]:
    result = {}
    for kw in keywords:
        total = 0
        for text in texts:
            total += text.lower().count(kw)
        if total > 0:
            result[kw] = total
    return dict(sorted(result.items(), key=lambda x: -x[1]))


texts_for_cat = [j.get("content", "") for j in filtered_jobs]

for category, keywords in CATEGORIES.items():
    counts = count_category(texts_for_cat, keywords)
    if counts:
        total = sum(counts.values())
        print(f"\n  {category} (загалом згадувань: {total}):")
        for kw, cnt in counts.items():
            bar = "#" * (cnt // 2)
            print(f"    {kw:<25} {cnt:>4}  {bar}")

# ---------------------------------------------------------------------------
# 14. Порівняння лематизації та стемінгу
# ---------------------------------------------------------------------------
print(f"\n{SEPARATOR}")
print("ПОРІВНЯННЯ: ЛЕМАТИЗАЦІЯ (SpaCy) vs СТЕМІНГ (NLTK)")
print(SEPARATOR)

sample_words = [
    "analytics",
    "analyzing",
    "analysis",
    "dashboards",
    "building",
    "reporting",
    "visualization",
    "optimization",
    "statistical",
    "segmentation",
    "recommendations",
    "communication",
    "processing",
    "proficiency",
    "monitoring",
    "engineering",
]

print(f"  {'Оригінал':<20} {'SpaCy Лема':<20} {'NLTK Лема':<20} {'Porter Стем':<20}")
print(f"  {'-' * 20} {'-' * 20} {'-' * 20} {'-' * 20}")
for word in sample_words:
    doc = nlp(word)
    spacy_lem = doc[0].lemma_ if doc else word
    nltk_lem = wnl.lemmatize(word)
    stem = stemmer.stem(word)
    print(f"  {word:<20} {spacy_lem:<20} {nltk_lem:<20} {stem:<20}")

# ---------------------------------------------------------------------------
# 15. Підсумок
# ---------------------------------------------------------------------------
print(f"\n{SEPARATOR}")
print("ВИСНОВКИ: ДОМІНАНТНІ ВИМОГИ ДО DATA ANALYST")
print(SEPARATOR)

top_tech = tech_counter.most_common(10)
top_lemmas = freq_spacy.most_common(15)

print(
    """
На основі аналізу {n} вакансій Data Analyst з Djinni.co виявлено
наступні домінантні вимоги:

1. КЛЮЧОВІ ТЕХНІЧНІ НАВИЧКИ:
""".format(n=len(filtered_jobs))
)

for i, (tech, cnt) in enumerate(top_tech[:10], 1):
    pct = round(cnt / len(filtered_jobs), 1)
    print(f"   {i:>2}. {tech:<25} ({cnt} згадувань, ~{pct} на вакансію)")

print("""
2. КЛЮЧОВІ КОМПЕТЕНЦІЇ (за лемами):
""")

# Фільтруємо лише значущі аналітичні терміни
MEANINGFUL_TERMS = {
    "datum",
    "data",
    "analysis",
    "report",
    "dashboard",
    "metric",
    "insight",
    "business",
    "product",
    "query",
    "visualization",
    "test",
    "performance",
    "funnel",
    "retention",
    "segment",
    "monitor",
    "model",
    "automation",
    "pipeline",
    "decision",
    "stakeholder",
    "conversion",
    "kpi",
}

meaningful_freq = {w: c for w, c in freq_spacy.items() if w in MEANINGFUL_TERMS}
meaningful_sorted = sorted(meaningful_freq.items(), key=lambda x: -x[1])

for i, (word, cnt) in enumerate(meaningful_sorted[:15], 1):
    print(f"   {i:>2}. {word:<25} ({cnt} згадувань)")

print()
