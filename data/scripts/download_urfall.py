"""
Download and inspect UR Fall Detection Dataset.
Source: https://fenix.ur.edu.pl/mkepski/ds/uf.html
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

DATA_DIR = Path("data/ur_fall")

# UR Fall download URLs (from the official page)
BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"

# Fall sequences: fall-01 to fall-30, camera 0 and camera 1
# ADL sequences: adl-01 to adl-40, camera 0 only
SEQUENCES = {
    "falls": [(f"fall-{i:02d}-cam0", f"{BASE_URL}/fall-{i:02d}-cam0-rgb.zip", f"{BASE_URL}/fall-{i:02d}-cam0-d.zip") for i in range(1, 31)] +
             [(f"fall-{i:02d}-cam1", f"{BASE_URL}/fall-{i:02d}-cam1-rgb.zip", f"{BASE_URL}/fall-{i:02d}-cam1-d.zip") for i in range(1, 31)],
    "adls": [(f"adl-{i:02d}-cam0", f"{BASE_URL}/adl-{i:02d}-cam0-rgb.zip", f"{BASE_URL}/adl-{i:02d}-cam0-d.zip") for i in range(1, 41)],
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, dest_path):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        return True
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest_path.name) as t:
            urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"  Error extracting {zip_path}: {e}")
        return False


def download_urfall(max_sequences=None):
    """
    Download UR Fall dataset.
    
    Args:
        max_sequences: Limit downloads for testing (None = download all)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_dir = DATA_DIR / "zips"
    zip_dir.mkdir(exist_ok=True)
    
    all_sequences = SEQUENCES["falls"] + SEQUENCES["adls"]
    
    if max_sequences:
        all_sequences = all_sequences[:max_sequences]
        print(f"Downloading first {max_sequences} sequences (test mode)")
    
    print(f"Downloading {len(all_sequences)} sequences...")
    print(f"Destination: {DATA_DIR}\n")
    
    successful = 0
    for seq_name, rgb_url, depth_url in tqdm(all_sequences, desc="Sequences"):
        seq_dir = DATA_DIR / seq_name
        
        # Skip if already extracted
        if (seq_dir / "rgb").exists() and (seq_dir / "depth").exists():
            successful += 1
            continue
        
        # Download RGB
        rgb_zip = zip_dir / f"{seq_name}-rgb.zip"
        if download_file(rgb_url, rgb_zip):
            extract_zip(rgb_zip, seq_dir)
        
        # Download Depth
        depth_zip = zip_dir / f"{seq_name}-d.zip"
        if download_file(depth_url, depth_zip):
            extract_zip(depth_zip, seq_dir)
        
        if (seq_dir / "rgb").exists() or (seq_dir / "depth").exists():
            successful += 1
    
    print(f"\nSuccessfully downloaded: {successful}/{len(all_sequences)} sequences")
    return successful


def reorganize_structure():
    """
    Reorganize extracted folders from:
        seq_dir/seq-name-rgb/*.png  →  seq_dir/rgb/*.png
        seq_dir/seq-name-d/*.png    →  seq_dir/depth/*.png
    """
    print("\nReorganizing directory structure...")
    
    import shutil
    
    reorganized = 0
    for seq_dir in DATA_DIR.iterdir():
        if not seq_dir.is_dir() or seq_dir.name == "zips":
            continue
        
        # Find the nested rgb folder (e.g., fall-01-cam0-rgb or adl-01-cam0-rgb)
        rgb_candidates = list(seq_dir.glob("*-rgb"))
        depth_candidates = list(seq_dir.glob("*-d"))
        
        moved = False
        
        # Handle RGB
        if rgb_candidates and not (seq_dir / "rgb").exists():
            src = rgb_candidates[0]
            dst = seq_dir / "rgb"
            shutil.move(str(src), str(dst))
            moved = True
        
        # Handle Depth
        if depth_candidates and not (seq_dir / "depth").exists():
            src = depth_candidates[0]
            dst = seq_dir / "depth"
            shutil.move(str(src), str(dst))
            moved = True
        
        if moved:
            reorganized += 1
    
    print(f"Reorganized {reorganized} sequences")
    

def inspect_urfall():
    """Inspect the downloaded dataset."""
    print("\n" + "="*70)
    print("UR Fall Dataset Inspection")
    print("="*70)
    
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    # Count sequences
    fall_dirs = sorted(DATA_DIR.glob("fall-*-cam*"))
    adl_dirs = sorted(DATA_DIR.glob("adl-*-cam*"))
    
    print(f"\nFound {len(fall_dirs)} fall sequences")
    print(f"Found {len(adl_dirs)} ADL sequences")
    
    # Detailed stats
    stats = {"fall": {"rgb_frames": 0, "depth_frames": 0, "sequences": 0},
             "adl": {"rgb_frames": 0, "depth_frames": 0, "sequences": 0}}
    
    for seq_dir in fall_dirs + adl_dirs:
        seq_type = "fall" if "fall" in seq_dir.name else "adl"
        
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        
        if rgb_dir.exists():
            rgb_frames = list(rgb_dir.glob("*.png"))
            stats[seq_type]["rgb_frames"] += len(rgb_frames)
        
        if depth_dir.exists():
            depth_frames = list(depth_dir.glob("*.png"))
            stats[seq_type]["depth_frames"] += len(depth_frames)
            stats[seq_type]["sequences"] += 1
    
    print(f"\n{'Category':<12} {'Sequences':<12} {'RGB Frames':<15} {'Depth Frames':<15}")
    print("-" * 55)
    print(f"{'Falls':<12} {stats['fall']['sequences']:<12} {stats['fall']['rgb_frames']:<15} {stats['fall']['depth_frames']:<15}")
    print(f"{'ADLs':<12} {stats['adl']['sequences']:<12} {stats['adl']['rgb_frames']:<15} {stats['adl']['depth_frames']:<15}")
    print("-" * 55)
    total_seq = stats['fall']['sequences'] + stats['adl']['sequences']
    total_rgb = stats['fall']['rgb_frames'] + stats['adl']['rgb_frames']
    total_depth = stats['fall']['depth_frames'] + stats['adl']['depth_frames']
    print(f"{'TOTAL':<12} {total_seq:<12} {total_rgb:<15} {total_depth:<15}")
    
    # Class balance
    fall_pct = 100 * stats['fall']['sequences'] / total_seq if total_seq > 0 else 0
    print(f"\nClass balance: {stats['fall']['sequences']}/{total_seq} fall sequences ({fall_pct:.1f}%)")


def inspect_sample_frames():
    """Load and inspect sample RGB and depth frames."""
    print("\n" + "="*70)
    print("Sample Frame Inspection")
    print("="*70)
    
    # Find first fall sequence with both RGB and depth
    for seq_dir in sorted(DATA_DIR.glob("fall-*-cam0")):
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        
        if not (rgb_dir.exists() and depth_dir.exists()):
            continue
        
        rgb_frames = sorted(rgb_dir.glob("*.png"))
        depth_frames = sorted(depth_dir.glob("*.png"))
        
        if not (rgb_frames and depth_frames):
            continue
        
        print(f"\nSample sequence: {seq_dir.name}")
        print(f"RGB frames: {len(rgb_frames)}, Depth frames: {len(depth_frames)}")
        
        # Inspect RGB
        rgb_img = cv2.imread(str(rgb_frames[0]))
        print(f"\nRGB frame: {rgb_frames[0].name}")
        print(f"  Shape: {rgb_img.shape}")
        print(f"  Dtype: {rgb_img.dtype}")
        
        # Inspect Depth (PNG16 format)
        depth_img = cv2.imread(str(depth_frames[0]), cv2.IMREAD_UNCHANGED)
        print(f"\nDepth frame: {depth_frames[0].name}")
        print(f"  Shape: {depth_img.shape}")
        print(f"  Dtype: {depth_img.dtype}")
        print(f"  Min: {depth_img.min()}, Max: {depth_img.max()}")
        
        if depth_img.dtype == np.uint16:
            # Kinect depth is typically in mm
            valid_mask = depth_img > 0
            if valid_mask.any():
                print(f"  Valid depth range: {depth_img[valid_mask].min()}mm - {depth_img[valid_mask].max()}mm")
                print(f"  Mean (valid): {depth_img[valid_mask].mean():.1f}mm")
                print(f"  Invalid pixels (depth=0): {(~valid_mask).sum()} ({100*(~valid_mask).sum()/depth_img.size:.1f}%)")
        
        return True
    
    print("No complete sequences found!")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Download only 3 sequences for testing")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect, don't download")
    parser.add_argument("--reorganize-only", action="store_true", help="Only reorganize existing downloads")
    args = parser.parse_args()
    
    if args.reorganize_only:
        reorganize_structure()
    elif not args.inspect_only:
        max_seq = 3 if args.test else None
        download_urfall(max_sequences=max_seq)
        reorganize_structure()
    
    inspect_urfall()
    inspect_sample_frames()