"""
Automated Data Downloader for UCI HAPT Dataset
Dataset: Smartphone-Based Recognition of Human Activities and Postural Transitions
Source: UCI Machine Learning Repository (ID: 341)
URL: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
"""

import os
import sys
import ssl
import zipfile
import urllib.request
import subprocess
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
ZIP_PATH = RAW_DATA_DIR / "hapt_dataset.zip"
EXTRACT_DIR = RAW_DATA_DIR / "HAPT_Data_Set"


def download_dataset():
    """Downloads and extracts the UCI HAPT dataset archive if not already extracted."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[INFO] Dataset already extracted at: {EXTRACT_DIR}")
        return

    if not ZIP_PATH.exists():
        print(f"[INFO] Downloading UCI HAPT dataset from:\n  {DATA_URL}")
        print(f"[INFO] Saving to: {ZIP_PATH}")
        try:
            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(
                DATA_URL,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            with urllib.request.urlopen(req, context=ctx) as response, open(ZIP_PATH, 'wb') as out_file:
                total_size = response.headers.get('Content-Length')
                total_size = int(total_size) if total_size else None
                bytes_downloaded = 0
                block_size = 64 * 1024
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size:
                        percent = (bytes_downloaded / total_size) * 100
                        mb = bytes_downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r[INFO] Progress: {mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
                        sys.stdout.flush()
            print("\n[SUCCESS] Download completed via urllib.")
        except Exception as e:
            print(f"\n[WARN] urllib download failed: {e}. Falling back to curl...")
            curl_cmd = ["curl", "-L", "-k", "-o", str(ZIP_PATH), DATA_URL]
            subprocess.run(curl_cmd, check=True)
            print("[SUCCESS] Download completed via curl.")

    print(f"[INFO] Extracting {ZIP_PATH.name} to {EXTRACT_DIR} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"[SUCCESS] Dataset successfully extracted to: {EXTRACT_DIR}")


if __name__ == "__main__":
    download_dataset()
