from bs4 import BeautifulSoup
import json
import random
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth


def scrape_with_playwright(url):
    # We wrap the Playwright initialization in Stealth().use_sync()
    with Stealth().use_sync(sync_playwright()) as p:
        # Launch browser (set headless=False the first time to watch it work)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print("Navigating to URL...")

        # Go to the URL and wait for the WAF JS challenge to finish
        page.goto(url, wait_until="networkidle")

        # Hard wait to ensure the JS redirect finishes
        page.wait_for_timeout(5000)

        print("Bypass complete. Current URL:", page.url)

        # --- Extract the AWS Token ---
        cookies = context.cookies()
        aws_token = next(
            (
                cookie["value"]
                for cookie in cookies
                if cookie["name"] == "aws-waf-token"
            ),
            None,
        )
        if aws_token:
            print(f"Success! AWS WAF Token acquired: {aws_token[:20]}...")

        html_content = page.content()
        browser.close()

        return html_content, aws_token


def extract_reviews_from_html(html_content: str, url: str):
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = soup.find_all("div", class_="CommentCard__comment")

    review_items = []

    for review in reviews:
        text_div = review.find("div", itemprop="reviewBody")
        if not text_div:
            continue
        text = text_div.get_text(strip=True)

        rating_block = review.select_one("div[class*='Rating__rating']")
        stars = 0
        if rating_block:
            stars = len(rating_block.select("span[class*='Rating__filled']"))

        review_items.append({"text": text, "stars": stars, "product": url})

    return review_items


def parse_reviews(url: str):
    print(f"Fetching page {url}")

    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(5000)

        all_reviews = []
        visited_pages = set()

        while True:
            html_content = page.content()
            all_reviews.extend(extract_reviews_from_html(html_content, url))

            active_page = page.locator(
                "ul[class*='Pagination__pages'] button[class*='Pagination__page'][class*='Pagination__isActive']"
            ).first

            if active_page.count() == 0:
                break

            current_page_text = active_page.inner_text().strip()
            if not current_page_text.isdigit():
                break

            current_page = int(current_page_text)
            if current_page in visited_pages:
                break
            visited_pages.add(current_page)

            next_page = current_page + 1
            page_buttons = page.locator(
                "ul[class*='Pagination__pages'] button[class*='Pagination__page']"
            )
            next_button = None

            for i in range(page_buttons.count()):
                text = page_buttons.nth(i).inner_text().strip()
                if text.isdigit() and int(text) == next_page:
                    next_button = page_buttons.nth(i)
                    break

            if next_button is None:
                break

            next_button.click()
            page.wait_for_function(
                """
                (targetPage) => {
                    const active = document.querySelector(
                        "ul[class*='Pagination__pages'] button[class*='Pagination__isActive']"
                    );
                    return active && active.textContent.trim() === String(targetPage);
                }
                """,
                arg=next_page,
            )
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(random.randint(900, 1500))

        browser.close()

    return all_reviews


if __name__ == "__main__":
    url = "https://makeup.com.ua/product/17050/"
    reviews = parse_reviews(url)
    print(f"Total articles found: {len(reviews)}\n")
    print("Examples:\n")
    for idx, item in enumerate(reviews[:3], 1):
        print(f"{idx}. [{item['text']}] {item['product']}")
        print("\n")

    with open("makeup_reviews(clinique).json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=4)
    print("Successfully saved to makeup_reviews(clinique).json")
