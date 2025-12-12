"""
Download the UP Fall Detection Dataset. Optimized for parallel downloading.
Source: https://sites.google.com/up.edu.mx/har-up/

This file is optimized version of https://github.com/jpnm561/HAR-UP/blob/master/DataBaseDownload/Downloader_pydrive.py
Usage: clone the official repository and replace the script with this one. check out controlls in main().
To set up, refer: https://github.com/jpnm561/HAR-UP/tree/master/DataBaseDownload
"""

# -*- coding: utf-8 -*-
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Requires: pip install PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pDrive_functions import fileFinder, fileDownloader
from createFolder import createFolder

"""
*******************************************************************************
1. ROBUST CONNECT FUNCTION (HPC READY)
*******************************************************************************
"""
def connect():
    gauth = GoogleAuth()
    
    # Try to load existing credentials
    gauth.LoadCredentialsFile("credentials.json")
    
    if gauth.credentials is None:
        # No creds -> Force Command Line Auth (Copy-paste URL)
        print("First time login: Copy the URL below into your laptop browser.")
        gauth.CommandLineAuth() 
        
    elif gauth.access_token_expired:
        # Expired -> Refresh
        print("Token expired. Refreshing...")
        gauth.Refresh()
        
    else:
        # Valid -> Authorize
        gauth.Authorize()
        
    # Save creds for next time so you don't have to login again
    gauth.SaveCredentialsFile("credentials.json")

    drive = GoogleDrive(gauth)
    return gauth, drive

"""
*******************************************************************************
2. SMART DOWNLOADER (CHECKS EXISTENCE)
*******************************************************************************
"""
def download_smart(path, f_name, p_id, drive):
    # Check if file already exists locally
    full_path = os.path.join(path, f_name)
    if os.path.exists(full_path):
        # Optional: Check file size > 0 to ensure it's not a corrupted empty file
        if os.path.getsize(full_path) > 0:
            print(f'[SKIP] {f_name} already exists.')
            return
        else:
            print(f'[RE-DOWNLOAD] {f_name} found but empty.')

    # Find file ID on Drive
    f_id = fileFinder(f_name, p_id, drive)
    
    if f_id:
        print(f'   [DOWNLOADING] {f_name}...')
        try:
            fileDownloader(f_name, f_id, path, drive)
            print(f'   [DONE] {f_name}')
        except Exception as e:
            print(f'   [ERROR] Failed to download {f_name}: {e}')
    else:
        print(f'   [MISSING] Could not find {f_name} on Drive')

"""
*******************************************************************************
3. PARALLEL WORKER (PROCESS ONE SUBJECT)
*******************************************************************************
"""
def process_subject(args):
    """
    Worker function to handle ONE subject entirely.
    This runs in its own thread.
    """
    subject_idx, n_act, n_trl, n_cam, cameras, parent_dir = args
    
    # Each thread needs its own Drive instance to be safe(ish)
    gauth, drive = connect()
    
    # Root Folder ID for UP-Fall
    p_id = '1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58'
    
    sub = 'Subject' + str(subject_idx)
    
    # Find Subject Folder
    s_id = fileFinder(sub, p_id, drive)
    if not s_id:
        print(f'[{sub}] Folder not found!')
        return

    # Loop Activities
    for j in range(n_act[0], n_act[1] + 1):
        act = 'Activity' + str(j)
        a_id = fileFinder(act, s_id, drive)
        
        if not a_id: continue

        # Loop Trials
        for k in range(n_trl[0], n_trl[1] + 1):
            trl = 'Trial' + str(k)
            t_id = fileFinder(trl, a_id, drive)
            
            if not t_id: continue

            # Prepare Local Folder
            path = os.path.join(parent_dir, sub, act, trl)
            createFolder(path)

            # Download Cameras (The Images)
            if cameras:
                for l in range(n_cam[0], n_cam[1] + 1):
                    cam_file = sub + act + trl + 'Camera' + str(l) + '.zip'
                    download_smart(path, cam_file, t_id, drive)

"""
*******************************************************************************
4. MAIN (THREAD MANAGER)
*******************************************************************************
"""
def main():
    # SETTINGS
    parent_dir = '..//..//data//up_fall//' # Use standard Linux paths; REPLACE based on placement of the repo
    n_sub = [1, 17]   # Subjects 1 to 17
    n_act = [1, 11]   # Activities 1 to 11
    n_trl = [1, 3]    # Trials 1 to 3
    n_cam = [1, 2]    # Cameras 1 to 2 (side view + front view)
    
    # Create valid list of subject numbers
    subjects = list(range(n_sub[0], n_sub[1] + 1))
    
    # Packaging arguments for the worker
    # We create a list of tasks, where each task is "Process Subject X"
    tasks = []
    for s in subjects:
        tasks.append((s, n_act, n_trl, n_cam, True, parent_dir))

    print(f"Starting Parallel Download for {len(subjects)} Subjects...")
    print(f"Saving to: {parent_dir}")
    
    # Max Workers = 4 is safe for Google Drive API. 
    # If you go higher (e.g. 8), you risk 403 Rate Limit errors.
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_subject, tasks)

    print("\nAll downloads finished.")

if __name__ == "__main__":
    main()
