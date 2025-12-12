import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 1. URL of the page containing the download links
BASE_URL = "https://sites.google.com/up.edu.mx/har-up/" 
OUTPUT_FOLDER = "UP_Fall_Dataset"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def download_files():
    print(f"Scanning {BASE_URL} for zip files...")
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all links ending in .zip
    links = soup.find_all('a', href=True)
    zip_links = [link['href'] for link in links if link['href'].endswith('.zip')]
    
    print(f"Found {len(zip_links)} zip files. Starting download...")

    for i, link in enumerate(zip_links):
        # Handle relative URLs
        full_url = urljoin(BASE_URL, link)
        filename = os.path.join(OUTPUT_FOLDER, link.split("/")[-1])
        
        print(f"[{i+1}/{len(zip_links)}] Downloading {filename}...")
        
        # Download the file
        try:
            r = requests.get(full_url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {full_url}: {e}")

    print("Download complete!")

if __name__ == "__main__":
    download_files()