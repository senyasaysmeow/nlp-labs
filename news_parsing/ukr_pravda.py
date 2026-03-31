from datetime import datetime, timedelta
import json
import requests
from bs4 import BeautifulSoup
import time
import os

dir_path = "article/"
os.makedirs(dir_path, exist_ok=True)


def fetch_article_text(url, headers):
    try:
        print(f"Parsing article {url}")
        response = requests.get(url, headers=headers)
        for attempt in range(5):
            if response.status_code != 429:
                break
            wait = 2**attempt
            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"[Failed to fetch article. Status: {response.status_code}]"

        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find(
            "div",
            class_=lambda x: bool(
                x and isinstance(x, str) and c in x for c in ["post_text", "news_text"]
            ),
        )

        if content_div:
            paragraphs = content_div.find_all("p")
            return "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

        return "[Content structure not found]"
    except Exception as e:
        return f"[Error fetching article: {str(e)}]"


def parse_ukr_pravda_by_date(url: str, date: datetime):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,uk;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    full_url = f"{url}/date_{date.strftime('%d%m%Y')}"
    print(f"Fetching page {full_url}")
    response = requests.get(full_url, headers=headers)
    for attempt in range(5):
        if response.status_code != 429:
            break
        wait = 2**attempt
        print(f"Rate limited. Waiting {wait} seconds...")
        time.sleep(wait)
        response = requests.get(full_url, headers=headers)
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

        content = fetch_article_text(link, headers)
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

        for category, base_urls in category_websites.items():
            if link.startswith(tuple(base_urls)):
                link_category = category

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

    news = []
    for i in range(9):
        news.extend(parse_ukr_pravda_by_date(url, date))
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

    with open(f"{dir_path}ukr_pravda_news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved to {dir_path}ukr_pravda_news.json")
