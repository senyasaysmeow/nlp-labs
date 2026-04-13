from datetime import datetime
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

        info_div = soup.select_one(".info > div:first-child")
        if info_div:
            text = info_div.get_text(strip=True)

            time_text, date_text, *_ = text.split()  # "17:56", "13.04.2026", "Пн"
            date = datetime.strptime(date_text, "%d.%m.%Y")

            pub_time = f"{date.strftime('%Y-%m-%d')}T{time_text}+02:00"

        content_div = soup.find("div", class_="txt")

        if content_div:
            paragraphs = content_div.find_all("p")
            return (
                "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()]),
                pub_time,
            )

        return "[Content structure not found]"
    except Exception as e:
        return f"[Error fetching article: {str(e)}]"


def parse_rbc_latest(url: str, scraper):
    print(f"Fetching page {url}")
    time.sleep(random.uniform(1, 2))  # Random delay
    response = scraper.get(url)
    for attempt in range(5):
        if response.status_code != 429:
            break
        wait = 2**attempt + random.uniform(0, 1)
        print(f"Rate limited. Waiting {wait:.1f} seconds...")
        time.sleep(wait)
        response = scraper.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="newsline")
    if content_div:
        articles = content_div.find_all("div")
    else:
        return "[Content structure not found]"

    news_items = []

    for article in articles:
        link_tag = article.find("a")
        if not link_tag:
            continue
        link_tag.span.decompose()
        title = link_tag.get_text(strip=True)
        link = link_tag.get("href")

        content, pub_time = fetch_article_text(link, scraper)

        news_items.append(
            {"title": title, "link": link, "time": pub_time, "content": content}
        )

    return news_items


if __name__ == "__main__":
    # Create cloudscraper session to bypass Cloudflare
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    news = []
    for i in range(1, 3):
        url = f"https://www.rbc.ua/rus/allnews/{i}"
        latest_news = parse_rbc_latest(url, scraper)
        news.extend(latest_news)
    print(f"Total articles found: {len(news)}\n")
    print("Examples:\n")
    for idx, item in enumerate(news[:3], 1):
        print(f"{idx}. [{item['time']}] {item['title']}")
        print(f"   Link: {item['link']}")
        print("\n--- CONTENT PREVIEW ---")

        preview = item["content"][:300].replace("\n", " ")
        print(f"{preview}{'...' if len(item['content']) > 300 else ''}")
        print("\n")

    with open(f"{dir_path}/rbc_news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved to {dir_path}/rbc_news.json")
