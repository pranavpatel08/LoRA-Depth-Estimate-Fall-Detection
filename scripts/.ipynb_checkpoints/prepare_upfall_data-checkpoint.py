"""
Unzips UP-Fall dataset files and standardizes directory structure.
Converts:
  .../Trial1/Subject1Activity1Trial1Camera1.zip
Into:
  .../Trial1/Camera1/ (containing images)
"""

import os
import zipfile
import re
from concurrent.futures import ThreadPoolExecutor

# CONFIGURATION
# ---------------------------------------------------------
ROOT_DIR = './data/up_fall' 

# Set to True to delete zip files after successful extraction
DELETE_ZIPS = True 
# ---------------------------------------------------------

def extract_worker(file_info):
    root, filename = file_info
    source_path = os.path.join(root, filename)
    
    # REGEX LOGIC:
    # Matches patterns like "Trial1Subject14Activity4Trial1Camera1.zip"
    # Group 1 = Trial Number (e.g., "1")
    # Group 2 = Camera Number (e.g., "1")
    # We look for "Trial" followed by digits, then later "Camera" followed by digits.
    match = re.search(r"Trial(\d+).*Camera(\d+)", filename, re.IGNORECASE)
    
    if not match:
        return f"[SKIP] Filename pattern mismatch: {filename}"
    
    trial_num = match.group(1)  # "1"
    cam_num = match.group(2)    # "1"
    
    # CONSTRUCT PATH: ./Subject/Activity/TrialX/CameraY
    # root is currently ".../SubjectX/ActivityY"
    trial_folder = f"Trial{trial_num}"
    cam_folder = f"Camera{cam_num}"
    
    # Final destination: .../ActivityY/Trial1/Camera1/
    dest_dir = os.path.join(root, trial_folder, cam_folder)
    
    try:
        # 1. Create the TrialX/CameraY folders
        os.makedirs(dest_dir, exist_ok=True)

        # 2. Extract contents directly into CameraY
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            # Check if already extracted (simple check for first file)
            file_list = zip_ref.namelist()
            if not file_list: 
                return f"[WARN] Empty Zip: {filename}"
                
            first_file = file_list[0]
            if os.path.exists(os.path.join(dest_dir, first_file)):
                return f"[EXISTS] Skipping {filename}"
            
            zip_ref.extractall(dest_dir)
            
        # 3. Cleanup (Optional)
        if DELETE_ZIPS:
            os.remove(source_path)
            return f"[DONE+DEL] {filename} -> {trial_folder}/{cam_folder}"
        
        return f"[DONE] {filename} -> {trial_folder}/{cam_folder}"

    except Exception as e:
        return f"[ERROR] {filename}: {str(e)}"

def main():
    print(f"Scanning '{ROOT_DIR}' for zip files...")
    
    tasks = []
    
    # Walk strictly through the directory structure
    for root, dirs, files in os.walk(ROOT_DIR):
        for filename in files:
            if filename.lower().endswith(".zip"):
                tasks.append((root, filename))
                
    if not tasks:
        print("No zip files found. Check your ROOT_DIR path.")
        return

    print(f"Found {len(tasks)} zip files. Extracting to ./TrialX/CameraY/ ...")
    
    # Parallel execution for speed on HPC
    with ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(extract_worker, tasks):
            print(result)

    print("\nExtraction Complete.")

if __name__ == "__main__":
    main()