"""
Download and extract NYU Depth V2 dataset.
"""

import os
import urllib.request
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Config
DATA_DIR = Path("data/nyu_depth_v2")
NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
MAT_FILE = DATA_DIR / "nyu_depth_v2_labeled.mat"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_nyu():
    """Download the NYU Depth V2 .mat file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if MAT_FILE.exists():
        print(f"File already exists: {MAT_FILE}")
        return
    
    print(f"Downloading NYU Depth V2 (~2.8GB)...")
    print(f"URL: {NYU_URL}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(NYU_URL, MAT_FILE, reporthook=t.update_to)
    
    print(f"Downloaded to: {MAT_FILE}")


def inspect_nyu():
    """Inspect the contents of the .mat file."""
    if not MAT_FILE.exists():
        print("Mat file not found. Run download first.")
        return
    
    print(f"\nInspecting: {MAT_FILE}")
    
    with h5py.File(MAT_FILE, 'r') as f:
        print(f"\nKeys in file: {list(f.keys())}")
        
        # The labeled dataset contains RGB images and depth maps
        if 'images' in f:
            images = f['images']
            print(f"\nImages shape: {images.shape}")
            print(f"Images dtype: {images.dtype}")
        
        if 'depths' in f:
            depths = f['depths']
            print(f"\nDepths shape: {depths.shape}")
            print(f"Depths dtype: {depths.dtype}")
            
            # Sample depth statistics
            sample_depth = np.array(depths[0])
            print(f"\nSample depth stats:")
            print(f"  Min: {sample_depth.min():.3f}m")
            print(f"  Max: {sample_depth.max():.3f}m")
            print(f"  Mean: {sample_depth.mean():.3f}m")


if __name__ == "__main__":
    download_nyu()
    inspect_nyu()