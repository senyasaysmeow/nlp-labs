import json
import requests
from bs4 import BeautifulSoup
import time
import os


dir_path = "jobs/"
os.makedirs(dir_path, exist_ok=True)


def fetch_job_text(url, headers):
    try:
        print(f"Parsing job {url}")
        response = requests.get(url, headers=headers)
        for attempt in range(5):
            if response.status_code != 429:
                break
            wait = 2**attempt
            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"[Failed to fetch page. Status: {response.status_code}]"

        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find("div", class_="vacancy-section")

        if content_div:
            paragraphs = content_div.find_all(["p", "li"])
            return "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

        return "[Content structure not found]"
    except Exception as e:
        return f"[Error fetching page: {str(e)}]"


def parse_dou_latest(url: str):
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
    jobs = soup.find_all("li", class_="l-vacancy")

    jobs_items = []

    for job in jobs:
        link_tag = job.find("a", class_="vt")
        if not link_tag:
            continue

        title = link_tag.text.strip()
        link = link_tag.get("href")

        time_tag = job.find("div", class_="date")
        if not time_tag:
            continue

        pub_time = time_tag.text.strip()
        content = fetch_job_text(link, headers)

        jobs_items.append(
            {"title": title, "link": link, "time": pub_time, "content": content}
        )

    return jobs_items


if __name__ == "__main__":
    # jobs = []
    url = "https://jobs.dou.ua/vacancies/?search=Data+Analyst"
    latest_jobs = parse_dou_latest(url)
    # news.extend(latest_news)
    print(f"Total offers found: {len(latest_jobs)}\n")
    print("Examples:\n")
    for idx, item in enumerate(latest_jobs[:3], 1):
        print(f"{idx}. [{item['time']}] {item['title']}")
        print(f"   Link: {item['link']}")
        print("\n--- CONTENT PREVIEW ---")

        preview = item["content"][:300].replace("\n", " ")
        print(f"{preview}{'...' if len(item['content']) > 300 else ''}")
        print("\n")

    with open(f"{dir_path}dou_jobs.json", "w", encoding="utf-8") as f:
        json.dump(latest_jobs, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved to {dir_path}dou_jobs.json")
