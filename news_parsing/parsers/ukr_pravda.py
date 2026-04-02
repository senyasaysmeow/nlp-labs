from datetime import datetime, timedelta
import json
import cloudscraper
from bs4 import BeautifulSoup
import time
import os
import random
from pathlib import Path


script_dir = Path(__file__).parent.parent
dir_path = script_dir / "articles/"
os.makedirs(dir_path, exist_ok=True)


def fetch_article_text(url, scraper):
    try:
        print(f"Parsing article {url}")
        time.sleep(random.uniform(1, 3))  # Random delay to avoid detection
        response = scraper.get(url)
        for attempt in range(5):
            if response.status_code != 429:
                break
            wait = 2**attempt + random.uniform(0, 1)
            print(f"Rate limited. Waiting {wait:.1f} seconds...")
            time.sleep(wait)
            response = scraper.get(url)
        if response.status_code != 200:
            return f"[Failed to fetch article. Status: {response.status_code}]"

        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find(
            "div",
            class_=lambda x: x and any(c in str(x) for c in ["post_text", "news_text"]),
        )

        if content_div:
            paragraphs = content_div.find_all("p")
            return "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

        return "[Content structure not found]"
    except Exception as e:
        return f"[Error fetching article: {str(e)}]"


def parse_ukr_pravda_by_date(url: str, date: datetime, scraper):
    full_url = f"{url}/date_{date.strftime('%d%m%Y')}"
    print(f"Fetching page {full_url}")
    time.sleep(random.uniform(1, 2))  # Random delay
    response = scraper.get(full_url)
    for attempt in range(5):
        if response.status_code != 429:
            break
        wait = 2**attempt + random.uniform(0, 1)
        print(f"Rate limited. Waiting {wait:.1f} seconds...")
        time.sleep(wait)
        response = scraper.get(full_url)
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    news_div = soup.find(
        "div", class_=lambda x: bool(x and isinstance(x, str) and "section_news" in x)
    )

    if not news_div:
        print("News section not found on page.")
        return []

    articles = news_div.find_all(
        "div", class_=lambda x: bool(x and isinstance(x, str) and "news_list" in x)
    )

    news_items = []

    for article in articles:
        link_tag = article.find("a")
        if not link_tag:
            continue

        title = "".join(
            t for t in link_tag.find_all(string=True, recursive=False)
        ).strip()
        link = link_tag.get("href")

        time_tag = article.find(
            "div",
            class_=lambda x: bool(x and isinstance(x, str) and "article_time" in x),
        )
        if not time_tag:
            continue
        time_text = time_tag.text.strip()
        pub_time = f"{date.strftime('%Y-%m-%d')}T{time_text}+02:00"

        content = fetch_article_text(link, scraper)
        if content == "[Content structure not found]":
            continue

        category_websites = {
            "politics": [
                "https://www.pravda.com.ua/",
                "https://www.eurointegration.com.ua/",
            ],
            "economics": ["https://epravda.com.ua/"],
            "life": ["https://life.pravda.com.ua/"],
            "technologies": ["https://mezha.ua/"],
            "defence": ["https://oboronka.mezha.ua/"],
            "sport": ["https://champion.com.ua/"],
        }

        link_category = "other"  # Default category
        for category, base_urls in category_websites.items():
            if link and link.startswith(tuple(base_urls)):
                link_category = category
                break

        news_items.append(
            {
                "title": title,
                "link": link,
                "time": pub_time,
                "content": content,
                "category": link_category,
            }
        )

    return news_items


if __name__ == "__main__":
    date = datetime.now()
    url = "https://pravda.com.ua/news"

    # Create cloudscraper session to bypass Cloudflare
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    news = []
    for i in range(2):
        news.extend(parse_ukr_pravda_by_date(url, date, scraper))
        date -= timedelta(days=1)
    print(f"Total articles found: {len(news)}\n")
    print("Examples:\n")
    for idx, item in enumerate(news[:3], 1):
        print(f"{idx}. [{item['time']}] {item['title']}")
        print(f"   Link: {item['link']}")
        print("\n--- CONTENT PREVIEW ---")

        preview = item["content"][:300].replace("\n", " ")
        print(f"{preview}{'...' if len(item['content']) > 300 else ''}")
        print("\n")

    with open(f"{dir_path}/ukr_pravda_news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved to {dir_path}/ukr_pravda_news.json")
