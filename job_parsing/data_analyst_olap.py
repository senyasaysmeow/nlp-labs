import json
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pylab

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE = os.path.join(SCRIPT_DIR, "jobs", "djinni_jobs.json")


def load_jobs_data(filepath: str) -> list:
    """Завантаження даних про вакансії з JSON файлу"""
    with open(filepath, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Завантажено {len(jobs)} вакансій")
    return jobs


def filter_data_analyst_jobs(jobs: list) -> list:
    """Фільтрація вакансій, пов'язаних з Data Analyst"""
    keywords = [
        "data analyst",
        "аналітик даних",
        "business analyst",
        "bi analyst",
        "product analyst",
        "marketing analyst",
        "analytics",
    ]

    filtered = []
    for job in jobs:
        title_lower = job["title"].lower()
        content_lower = job.get("content", "").lower()

        # Перевіряємо наявність ключових слів у заголовку або описі
        for keyword in keywords:
            if keyword in title_lower or keyword in content_lower:
                filtered.append(job)
                break

    print(f"Знайдено {len(filtered)} вакансій для Data Analyst")
    return filtered


def extract_experience_years(content: str) -> int:
    """Витягування вимог до досвіду (у роках)"""
    patterns = [
        r"(\d+)\+?\s*(?:years?|років|роки|рік)",
        r"(\d+)\+?\s*(?:years? of experience|років досвіду)",
        r"experience[:\s]+(\d+)\+?\s*(?:years?|років)",
        r"(\d+)-(\d+)\s*(?:years?|років)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content.lower())
        if match:
            years = int(match.group(1))
            return min(years, 10)  # Обмежуємо максимум 10 років
    return 0  # Якщо не вказано - вважаємо junior (0 років)


def extract_technologies(content: str) -> list:
    """Витягування технологій з опису вакансії"""
    tech_keywords = {
        "SQL": ["sql", "postgresql", "mysql", "oracle", "bigquery"],
        "Python": ["python", "pandas", "numpy", "scipy"],
        "Excel": ["excel", "spreadsheet"],
        "Tableau": ["tableau"],
        "Power BI": ["power bi", "powerbi"],
        "R": [" r ", " r,", "r language", "rstudio"],
        "Spark": ["spark", "pyspark"],
        "Airflow": ["airflow"],
        "Looker": ["looker"],
        "ClickHouse": ["clickhouse"],
        "dbt": ["dbt"],
        "ETL": ["etl", "elt"],
    }

    content_lower = content.lower()
    found_tech = []

    for tech, keywords in tech_keywords.items():
        for keyword in keywords:
            if keyword in content_lower:
                found_tech.append(tech)
                break

    return found_tech


def extract_job_level(title: str, content: str) -> str:
    """Визначення рівня позиції"""
    text = (title + " " + content).lower()

    if "lead" in text or "head" in text or "principal" in text:
        return "Lead/Head"
    elif "senior" in text or "sr." in text or "5+" in text:
        return "Senior"
    elif "middle" in text or "mid" in text or "2-3" in text or "3+" in text:
        return "Middle"
    elif "junior" in text or "jr." in text or "entry" in text or "trainee" in text:
        return "Junior"
    else:
        return "Not Specified"


def analyze_jobs(jobs: list) -> dict:
    """Комплексний аналіз вакансій"""
    analysis = {
        "total_jobs": len(jobs),
        "experience_distribution": Counter(),
        "technology_distribution": Counter(),
        "level_distribution": Counter(),
        "jobs_by_date": Counter(),
        "detailed_jobs": [],
    }

    for job in jobs:
        content = job.get("content", "")
        title = job.get("title", "")
        time_str = job.get("time", "")

        # Досвід
        exp_years = extract_experience_years(content)
        analysis["experience_distribution"][exp_years] += 1

        # Технології
        technologies = extract_technologies(content)
        for tech in technologies:
            analysis["technology_distribution"][tech] += 1

        # Рівень
        level = extract_job_level(title, content)
        analysis["level_distribution"][level] += 1

        # Дата публікації
        date_part = time_str.split()[1] if " " in time_str else time_str
        analysis["jobs_by_date"][date_part] += 1

        # Детальна інформація
        analysis["detailed_jobs"].append(
            {
                "title": title,
                "experience_years": exp_years,
                "technologies": technologies,
                "level": level,
                "time": time_str,
            }
        )

    return analysis


def print_analysis_summary(analysis: dict):
    """Виведення підсумку аналізу"""
    print("\n" + "=" * 60)
    print("АНАЛІЗ РИНКУ ПРАЦІ ДЛЯ DATA ANALYST")
    print("=" * 60)

    print(f"\nЗагальна кількість вакансій: {analysis['total_jobs']}")

    print("\n--- Розподіл за рівнем позиції ---")
    for level, count in analysis["level_distribution"].most_common():
        percentage = (count / analysis["total_jobs"]) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")

    print("\n--- Топ-10 технологій ---")
    for tech, count in analysis["technology_distribution"].most_common(10):
        percentage = (count / analysis["total_jobs"]) * 100
        print(f"  {tech}: {count} ({percentage:.1f}%)")

    print("\n--- Вимоги до досвіду (роки) ---")
    for years in sorted(analysis["experience_distribution"].keys()):
        count = analysis["experience_distribution"][years]
        percentage = (count / analysis["total_jobs"]) * 100
        label = f"{years}+ років" if years > 0 else "Не вказано / Junior"
        print(f"  {label}: {count} ({percentage:.1f}%)")


def olap_cube_visualization(analysis: dict):
    """3D OLAP-візуалізація аналізу вакансій"""

    # Підготовка даних для візуалізації
    tech_counts = analysis["technology_distribution"].most_common(9)

    # Створюємо матрицю: рівні x технології
    levels = ["Junior", "Middle", "Senior", "Lead/Head", "Not Specified"]
    technologies = (
        [t[0] for t in tech_counts] if tech_counts else ["SQL", "Python", "Excel"]
    )

    # Заповнюємо матрицю реальними даними
    matrix = np.zeros((len(levels), len(technologies)))

    for job in analysis["detailed_jobs"]:
        level_idx = levels.index(job["level"]) if job["level"] in levels else 4
        for tech in job["technologies"]:
            if tech in technologies:
                tech_idx = technologies.index(tech)
                matrix[level_idx, tech_idx] += 1

    # Створення 3D візуалізації
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Кольори для різних рівнів
    colors = ["#4bb2c5", "#c5b47f", "#EAA228", "#579575", "#839557"]

    xpos = np.arange(len(technologies))
    ypos = np.arange(len(levels))
    xposM, yposM = np.meshgrid(xpos, ypos, indexing="ij")

    xposM = xposM.flatten()
    yposM = yposM.flatten()
    zpos = np.zeros_like(xposM)

    dx = dy = 0.6
    dz = matrix.T.flatten()

    # Визначаємо кольори для кожного стовпця
    bar_colors = []
    for _ in range(len(technologies)):
        for j in range(len(levels)):
            bar_colors.append(colors[j % len(colors)])

    ax.bar3d(xposM, yposM, zpos, dx, dy, dz, color=bar_colors, alpha=0.8)

    ax.set_xlabel("\nТехнології", fontsize=10, labelpad=10)
    ax.set_ylabel("\nРівень позиції", fontsize=10, labelpad=10)
    ax.set_zlabel("\nКількість вакансій", fontsize=10, labelpad=10)

    ax.set_xticks(xpos + dx / 2)
    ax.set_xticklabels(technologies, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ypos + dy / 2)
    ax.set_yticklabels(levels, fontsize=8)

    ax.set_title(
        "OLAP-куб: Аналіз вакансій Data Analyst\n(Технології × Рівень × Кількість)",
        fontsize=12,
        fontweight="bold",
    )

    pylab.tight_layout()
    pylab.show()


def olap_experience_technology(analysis: dict):
    """3D OLAP: Досвід × Технології × Кількість"""

    tech_counts = analysis["technology_distribution"].most_common(9)
    technologies = (
        [t[0] for t in tech_counts] if tech_counts else ["SQL", "Python", "Excel"]
    )

    # Досвід: 0, 1-2, 3-4, 5+
    exp_ranges = ["0 (Junior)", "1-2 років", "3-4 років", "5+ років"]

    def get_exp_range(years):
        if years == 0:
            return 0
        elif years <= 2:
            return 1
        elif years <= 4:
            return 2
        else:
            return 3

    # Заповнюємо матрицю
    matrix = np.zeros((len(exp_ranges), len(technologies)))

    for job in analysis["detailed_jobs"]:
        exp_idx = get_exp_range(job["experience_years"])
        for tech in job["technologies"]:
            if tech in technologies:
                tech_idx = technologies.index(tech)
                matrix[exp_idx, tech_idx] += 1

    # Візуалізація
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["#4bb2c5", "#EAA228", "#579575", "#953579"]

    xpos = np.arange(len(technologies))
    ypos = np.arange(len(exp_ranges))

    for i, exp_range in enumerate(exp_ranges):
        heights = matrix[i, :]
        ax.bar(
            xpos, heights, zs=i, zdir="y", color=colors[i], alpha=0.8, label=exp_range
        )

    ax.set_xlabel("\nТехнології", fontsize=10, labelpad=10)
    ax.set_ylabel("\nДосвід", fontsize=10, labelpad=10)
    ax.set_zlabel("\nКількість вакансій", fontsize=10, labelpad=10)

    ax.set_xticks(xpos)
    ax.set_xticklabels(technologies, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(exp_ranges, fontsize=8)

    ax.set_title(
        "OLAP-куб: Досвід × Технології × Кількість вакансій",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)

    pylab.tight_layout()
    pylab.show()


def olap_summary_charts(analysis: dict):
    """Комплексна 2D візуалізація для огляду"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Аналіз ринку праці Data Analyst - OLAP Dashboard",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Розподіл за технологіями (барчарт)
    ax1 = axes[0, 0]
    tech_data = analysis["technology_distribution"].most_common(10)
    techs, counts = zip(*tech_data)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(techs)))
    bars = ax1.barh(techs, counts, color=colors)
    ax1.set_xlabel("Кількість вакансій")
    ax1.set_title("Топ-10 технологій")
    ax1.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    # 2. Розподіл за рівнем (pie chart)
    ax2 = axes[0, 1]
    level_data = analysis["level_distribution"]
    labels = list(level_data.keys())
    sizes = list(level_data.values())
    colors = ["#4bb2c5", "#c5b47f", "#EAA228", "#579575", "#953579"]
    explode = [0.05] * len(labels)
    ax2.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors[: len(labels)],
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax2.set_title("Розподіл за рівнем позиції")

    # 3. Розподіл за досвідом (барчарт)
    ax3 = axes[1, 0]
    exp_data = sorted(analysis["experience_distribution"].items())
    years = [f"{y}+ років" if y > 0 else "Junior" for y, _ in exp_data]
    counts = [c for _, c in exp_data]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(years)))
    bars = ax3.bar(years, counts, color=colors)
    ax3.set_xlabel("Вимоги до досвіду")
    ax3.set_ylabel("Кількість вакансій")
    ax3.set_title("Розподіл за вимогами до досвіду")
    ax3.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            fontsize=9,
        )

    # 4. Статистика (текст)
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_text = f"""
    СТАТИСТИКА РИНКУ ПРАЦІ
    
    Загальна кількість вакансій: {analysis["total_jobs"]}
    
    Найпопулярніші технології:
    """
    for tech, count in analysis["technology_distribution"].most_common(5):
        pct = (count / analysis["total_jobs"]) * 100
        stats_text += f"\n    • {tech}: {count} ({pct:.1f}%)"

    stats_text += "\n\n    Розподіл за рівнем:"
    for level, count in analysis["level_distribution"].most_common():
        pct = (count / analysis["total_jobs"]) * 100
        stats_text += f"\n    • {level}: {count} ({pct:.1f}%)"

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ПОШУК ВАКАНСІЙ АНАЛІТИКА ДАНИХ НА РИНКУ ПРАЦІ")
    print("Джерело даних: djinni.co")
    print("=" * 60)

    # Завантаження даних
    jobs = load_jobs_data(JOBS_FILE)

    # Фільтрація вакансій Data Analyst
    data_analyst_jobs = filter_data_analyst_jobs(jobs)

    # Аналіз вакансій
    analysis = analyze_jobs(data_analyst_jobs)

    # Виведення результатів
    print_analysis_summary(analysis)

    # OLAP-візуалізація
    print("\n" + "=" * 60)
    print("OLAP-ВІЗУАЛІЗАЦІЯ")
    print("=" * 60)
    print("\nОберіть тип візуалізації:")
    print("1 - 3D OLAP-куб: Технології × Рівень × Кількість")
    print("2 - 3D OLAP-куб: Досвід × Технології × Кількість")
    print("3 - Комплексний 2D Dashboard")
    print("4 - Показати всі візуалізації")
    print("5 - Вийти без візуалізації")

    try:
        choice = int(input("\nВаш вибір (1-5): "))
    except ValueError:
        choice = 3  # За замовчуванням - 2D Dashboard

    if choice == 1:
        olap_cube_visualization(analysis)
    elif choice == 2:
        olap_experience_technology(analysis)
    elif choice == 3:
        olap_summary_charts(analysis)
    elif choice == 4:
        olap_summary_charts(analysis)
        olap_cube_visualization(analysis)
        olap_experience_technology(analysis)
    elif choice == 5:
        print("Завершення роботи.")
        sys.exit(0)
    else:
        print("Невірний вибір. Показую 2D Dashboard.")
        olap_summary_charts(analysis)
