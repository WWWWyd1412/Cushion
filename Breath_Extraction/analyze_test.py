import os
import sys
import time
import numpy as np

# Add project root and Breath_Extraction to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader
import preprocess
import algorithms
from algorithms import base

def analyze():
    data_path = "../data/20260501_162541.txt"
    if not os.path.exists(data_path):
        data_path = "data/20260501_162541.txt"
    
    print(f"[{time.strftime('%H:%M:%S')}] Loading data from {data_path}...")
    raw_times, raw_frames = data_loader.load_pressure_txt(data_path)
    if raw_frames is None:
        print("Failed to load data.")
        return
        
    print(f"[{time.strftime('%H:%M:%S')}] Preprocessing data...")
    clean_times, clean_frames = preprocess.clean_dataset(raw_times, raw_frames)
    print(f"Original frames: {len(raw_frames)}, Clean frames: {len(clean_frames)}, Dropped: {len(raw_frames) - len(clean_frames)}")
    
    fs = 10.0
    
    # 1. Inspect ROI signal characteristics
    print("\n--- ROI Detection & Signal Extraction ---")
    try:
        # get_dual_roi_mean inside base.py
        roi_signal = base.get_dual_roi_mean(clean_frames, window_size=5)
        print(f"Extracted ROI 1D signal length: {len(roi_signal)}")
        print(f"ROI Signal Stats - Min: {np.min(roi_signal):.4f}, Max: {np.max(roi_signal):.4f}, Std: {np.std(roi_signal):.4f}")
    except Exception as e:
        print(f"Failed to extract ROI signal: {e}")
        roi_signal = None

    # 2. Run sliding window comparison
    win_size = 250
    step_size = 50
    total_frames = len(clean_frames)
    
    windows = []
    start_idx = 0
    while start_idx + win_size <= total_frames:
        windows.append((start_idx, start_idx + win_size))
        start_idx += step_size
        
    print(f"\n--- Comparing Algorithms (Sliding Window: {len(windows)} windows) ---")
    
    algos = {
        "EMD": algorithms.extract_emd,
        "VMD": algorithms.extract_vmd,
        "AFD": algorithms.extract_afd,
        "VMD_FPR": algorithms.extract_vmd_fpr,
        "SMVMD": algorithms.extract_smvmd,
        "MVMD": algorithms.extract_mvmd,
        "Multi-ROI ICA": algorithms.extract_multi_roi_ica
    }
    
    for name, func in algos.items():
        print(f"\nProcessing {name} in sliding windows...")
        bpms_peak = []
        bpms_fpr = []
        times = []
        
        for idx, (start, end) in enumerate(windows):
            window_frames = clean_frames[start:end]
            start_t = time.time()
            try:
                res = func(window_frames, fs)
                
                times.append(time.time() - start_t)
                
                # Apply smoothing
                smoothed = base.smooth_respiration_signal(res)
                
                # Align phase if not first window
                # (Skip phase alignment for simple BPM calculation)
                
                bpm_peak = base.calculate_bpm(smoothed, fs)
                bpm_fpr = base.calculate_bpm_fpr(smoothed, fs)
                
                bpms_peak.append(bpm_peak)
                bpms_fpr.append(bpm_fpr)
                
            except Exception as e:
                print(f"  Error in window {idx} ({start}-{end}): {e}")
                
        if len(times) > 0:
            avg_time = np.mean(times)
            print(f"[{name}] Window execution time: Mean = {avg_time:.3f}s, Total = {np.sum(times):.2f}s")
            
            # Filter out 0.0 BPMs for statistics
            bpms_peak_valid = [b for b in bpms_peak if b > 0.0]
            bpms_fpr_valid = [b for b in bpms_fpr if b > 0.0]
            
            print(f"      Peak BPM - Mean: {np.mean(bpms_peak_valid) if bpms_peak_valid else 0.0:.2f}, Min: {np.min(bpms_peak_valid) if bpms_peak_valid else 0.0:.2f}, Max: {np.max(bpms_peak_valid) if bpms_peak_valid else 0.0:.2f}, Std: {np.std(bpms_peak_valid) if bpms_peak_valid else 0.0:.2f} (Valid windows: {len(bpms_peak_valid)}/{len(bpms_peak)})")
            print(f"      FPR BPM  - Mean: {np.mean(bpms_fpr_valid) if bpms_fpr_valid else 0.0:.2f}, Min: {np.min(bpms_fpr_valid) if bpms_fpr_valid else 0.0:.2f}, Max: {np.max(bpms_fpr_valid) if bpms_fpr_valid else 0.0:.2f}, Std: {np.std(bpms_fpr_valid) if bpms_fpr_valid else 0.0:.2f} (Valid windows: {len(bpms_fpr_valid)}/{len(bpms_fpr)})")
        else:
            print(f"[{name}] Failed to run sliding windows.")

if __name__ == "__main__":
    analyze()
