"""
Prepare Le2i Fall Dataset: extract frames from videos.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def extract_frames(video_path: Path, output_dir: Path, fps: int = 10):
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Target frames per second
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  Warning: Could not open {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps <= 0:
        video_fps = 30  # Default assumption
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def main():
    # Adjust this path based on your extraction
    le2i_root = Path("data/le2i")
    
    # Find the actual data location (Kaggle extracts may have nested folders)
    possible_paths = [
        le2i_root,
        le2i_root / "FallDataset",
        le2i_root / "falldataset-imvia",
        le2i_root / "Fall Detection Dataset",
    ]
    
    data_root = None
    for p in possible_paths:
        if p.exists():
            # Look for video files or expected structure
            videos = list(p.rglob("*.avi")) # + list(p.rglob("*.mp4"))
            if videos:
                data_root = p
                break
    
    if data_root is None:
        print("Could not find Le2i dataset. Please check the path.")
        print(f"Searched in: {possible_paths}")
        print("\nListing contents of data/le2i:")
        if le2i_root.exists():
            for item in le2i_root.rglob("*"):
                print(f"  {item}")
        return
    
    print(f"Found data at: {data_root}")
    
    # Output directory
    output_root = Path("data/le2i_frames")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    videos = list(data_root.rglob("*.avi")) # + list(data_root.rglob("*.mp4"))
    print(f"Found {len(videos)} videos")
    
    # Categorize videos
    fall_videos = []
    adl_videos = []
    
    for v in videos:
        # Le2i naming convention usually has "fall" or "adl" in path/name
        # v_lower = str(v).lower()
        # if "fall" in v_lower and "not" not in v_lower:
        #     fall_videos.append(v)
        # elif "adl" in v_lower or "notfall" in v_lower or "daily" in v_lower or "coffee" in v_lower:
        #     adl_videos.append(v)
        # else:
        #     # Default classification based on common patterns
        #     if "chute" in v_lower:  # French for "fall"
        #         fall_videos.append(v)
        #     else:
        #         adl_videos.append(v)
        fall_videos.append(v)
    
    print(f"Fall videos: {len(fall_videos)}")
    print(f"ADL videos: {len(adl_videos)}")
    
    # Process videos
    print("\nExtracting frames from fall videos...")
    for i, video_path in enumerate(tqdm(fall_videos)):
        seq_name = f"fall_{i:03d}"
        output_dir = output_root / seq_name / "rgb"
        
        if output_dir.exists() and len(list(output_dir.glob("*.png"))) > 0:
            continue  # Skip if already processed
        
        n_frames = extract_frames(video_path, output_dir, fps=10)
        
        # Create label file
        with open(output_root / seq_name / "label.txt", 'w') as f:
            f.write("fall\n")
            f.write(f"source: {video_path.name}\n")
            f.write(f"frames: {n_frames}\n")
    
    print("\nExtracting frames from ADL videos...")
    for i, video_path in enumerate(tqdm(adl_videos)):
        seq_name = f"adl_{i:03d}"
        output_dir = output_root / seq_name / "rgb"
        
        if output_dir.exists() and len(list(output_dir.glob("*.png"))) > 0:
            continue
        
        n_frames = extract_frames(video_path, output_dir, fps=10)
        
        with open(output_root / seq_name / "label.txt", 'w') as f:
            f.write("adl\n")
            f.write(f"source: {video_path.name}\n")
            f.write(f"frames: {n_frames}\n")
    
    # Summary
    print("\n" + "="*60)
    print("Le2i Dataset Preparation Complete")
    print("="*60)
    
    fall_seqs = list(output_root.glob("fall_*"))
    adl_seqs = list(output_root.glob("adl_*"))
    
    print(f"Fall sequences: {len(fall_seqs)}")
    print(f"ADL sequences: {len(adl_seqs)}")
    
    total_frames = sum(
        len(list((s / "rgb").glob("*.png"))) 
        for s in fall_seqs + adl_seqs 
        if (s / "rgb").exists()
    )
    print(f"Total frames: {total_frames}")
    print(f"\nOutput saved to: {output_root}")


if __name__ == "__main__":
    main()