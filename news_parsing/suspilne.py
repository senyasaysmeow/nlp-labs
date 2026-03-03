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
            class_=lambda x: bool(x and isinstance(x, str) and "article-content" in x),
        )

        if content_div:
            paragraphs = content_div.find_all("p")
            return "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

        return "[Content structure not found]"
    except Exception as e:
        return f"[Error fetching article: {str(e)}]"


def parse_suspilne_latest(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,uk;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    print(f"Fetching page {url}")
    response = requests.get(url, headers=headers)
    for attempt in range(5):
        if response.status_code != 429:
            break
        wait = 2**attempt
        print(f"Rate limited. Waiting {wait} seconds...")
        time.sleep(wait)
        response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article")

    news_items = []

    for article in articles:
        link_tag = article.find(
            "a", class_=lambda x: bool(x and isinstance(x, str) and "headline" in x)
        )
        if not link_tag:
            continue

        title_span = link_tag.find(
            "span", class_=lambda x: bool(x and isinstance(x, str) and "label" not in x)
        )
        if not title_span:
            continue
        title = title_span.text.strip()
        link = link_tag.get("href")

        time_tag = article.find("time")
        if not time_tag:
            continue
        pub_time = time_tag.get("datetime")

        content = fetch_article_text(link, headers)

        news_items.append(
            {"title": title, "link": link, "time": pub_time, "content": content}
        )

    return news_items


if __name__ == "__main__":
    news = []
    for i in range(1, 11):
        url = f"https://suspilne.media/latest/?page={i}"
        latest_news = parse_suspilne_latest(url)
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

    with open(f"{dir_path}suspilne_news.json", "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved to {dir_path}suspilne_news.json")
