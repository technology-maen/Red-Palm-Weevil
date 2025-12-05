import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from urllib.parse import urljoin, urlparse

# --- CONFIG ---
BASE_URL = input("Enter Nursry URL: ")  # replace with actual URL
START_PAGE = "/" + input("Enter startpage: ")                   # replace with actual listing path
OUTPUT_CSV = "dataset.csv"
IMAGE_DIR = "images"

os.makedirs(IMAGE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "PalmAgeProjectBot/1.0 (+your_email@example.com)"
}

REQUEST_DELAY = 1.5  # seconds

# --- FUNCTIONS ---
def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def download_image(url, dest_path):
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=15)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def clean_text(txt):
    return txt.strip().replace("\n", " ").replace("\r", " ")

def parse_listing_page(page_url):
    soup = get_soup(page_url)
    items = []
    for card in soup.select("div.product-card"):  # adjust selector for the site
        try:
            img_tag = card.select_one("product-image-photo")
            img_url = urljoin(page_url, img_tag["src"])
            print(img_url)
            species = clean_text(card.select_one(".species-name").text)
            age_info = clean_text(card.select_one(".age-info").text)
            height_info = None
            h = card.select_one(".height-info")
            if h:
                height_info = clean_text(h.text)
            items.append({
                "image_url": img_url,
                "species": species,
                "age_info": age_info,
                "height_info": height_info,
                "source_url": page_url
            })
        except Exception as e:
            print("Failed to parse card:", e)
    return items

def scrape_nursery(start_url):
    data = []
    next_page = start_url
    while next_page:
        full_url = urljoin(BASE_URL, next_page)
        print("Scraping:", full_url)
        items = parse_listing_page(full_url)
        data.extend(items)
        time.sleep(REQUEST_DELAY)
        soup = get_soup(full_url)
        next_link = soup.select_one("a.next")  # adjust selector for "next page"
        next_page = next_link["href"] if next_link else None
    return data

def main():
    all_data = scrape_nursery(START_PAGE)
    print(all_data)
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Metadata saved to", OUTPUT_CSV)

    # Download images
    for i, row in df.iterrows():
        img_url = row["image_url"]
        parsed = urlparse(img_url)
        print(parsed)
        ext = os.path.splitext(parsed.path)[1]
        filename = f"{i:05d}{ext}"
        dest = os.path.join(IMAGE_DIR, filename)
        try:
            print(img_url)
            download_image(img_url, dest)
            df.at[i, "image_path"] = dest
        except Exception as e:
            print("Failed to download:", img_url, e)
        time.sleep(REQUEST_DELAY)

    df.to_csv(OUTPUT_CSV, index=False)
    print("Dataset ready! Images and metadata saved.")

if __name__ == "__main__":
    main()
