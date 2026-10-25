"""Tremor & Movement Signal Analysis for PARKNEROO.
Measures:
  - Signal amplitude (RMS, Peak-to-Peak, standard deviation)
  - Frequency components (FFT, Welch PSD, dominant peak frequency, Parkinsonian band power)
  - Movement variability (spectral entropy, tremor regularity index)
  - Estimated movement features labeling and session comparison
Libraries: SciPy, NumPy, Matplotlib, OpenCV, MediaPipe.

Disclaimer: A phone sensor or video does not capture every tremor reliably;
label results as estimated movement features.
"""
import io
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.stats import entropy
import mediapipe as mp

mp_hands = mp.solutions.hands

def bandpass_filter(data, lowcut=1.5, highcut=15.0, fs=30.0, order=4):
    """Zero-phase Butterworth bandpass filter to isolate tremor oscillations."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    if low >= high:
        return data - np.mean(data)
    b, a = butter(order, [low, high], btype='band')
    padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
    if padlen < 9:
        return data - np.mean(data)
    return filtfilt(b, a, data, padlen=padlen)


def analyze_tremor_signal(time_arr: np.ndarray, signal_arr: np.ndarray, source_label: str = "sensor"):
    """
    Analyzes a 1D or magnitude movement signal (acceleration or displacement).
    
    Returns:
      feats (dict): clinical tremor metrics
      psd_data (dict): frequency and power spectral density for plotting
      fig (matplotlib.figure.Figure): multi-panel clinical report plot
    """
    time_arr = np.asarray(time_arr, dtype=float)
    signal_arr = np.asarray(signal_arr, dtype=float)
    
    if len(signal_arr) < 30:
        raise ValueError("Insufficient signal duration (at least 1-2 seconds required).")
        
    # Estimate sampling rate
    dt_vals = np.diff(time_arr)
    median_dt = float(np.median(dt_vals)) if len(dt_vals) > 0 else 1.0 / 30.0
    fs = 1.0 / median_dt if median_dt > 0 else 30.0
    fs = float(np.clip(fs, 10.0, 200.0))
    
    # Detrend and filter
    detrended = detrend(signal_arr)
    filtered = bandpass_filter(detrended, lowcut=1.5, highcut=min(15.0, fs * 0.45), fs=fs)
    
    # 1. Signal Amplitude Metrics
    rms_amplitude = float(np.sqrt(np.mean(filtered**2)))
    peak_to_peak = float(np.ptp(filtered))
    std_amplitude = float(np.std(filtered, ddof=1))
    
    # 2. Frequency Components via Welch PSD
    nperseg = min(len(filtered), int(fs * 2.0))
    if nperseg < 16:
        nperseg = len(filtered)
    freqs, psd = welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    
    # Focus on clinical band (1.5 Hz to 14.0 Hz)
    valid_mask = (freqs >= 1.5) & (freqs <= 14.0)
    if np.any(valid_mask):
        sub_freqs = freqs[valid_mask]
        sub_psd = psd[valid_mask]
        peak_idx = np.argmax(sub_psd)
        dominant_freq = float(sub_freqs[peak_idx])
        peak_power = float(sub_psd[peak_idx])
    else:
        dominant_freq = 0.0
        peak_power = 0.0
        
    # Energy in Parkinsonian resting tremor band (3.5 - 7.5 Hz)
    pd_band = (freqs >= 3.5) & (freqs <= 7.5)
    pd_power = float(np.trapz(psd[pd_band], freqs[pd_band])) if np.any(pd_band) else 0.0
    
    # Energy in Physiological tremor band (8.0 - 12.0 Hz)
    phys_band = (freqs >= 8.0) & (freqs <= 12.0)
    phys_power = float(np.trapz(psd[phys_band], freqs[phys_band])) if np.any(phys_band) else 0.0
    
    total_power = float(np.trapz(psd[valid_mask], freqs[valid_mask])) if np.any(valid_mask) else 1e-9
    pd_power_ratio = float((pd_power / max(total_power, 1e-9)) * 100.0)
    
    # 3. Movement Variability & Regularity
    # Normalized spectral distribution
    norm_psd = sub_psd / (np.sum(sub_psd) + 1e-9) if np.any(valid_mask) else np.array([1.0])
    spec_entropy = float(entropy(norm_psd)) if len(norm_psd) > 1 else 0.0
    # Regularity index: fraction of power contained in peak bin +/- 1 bin
    if np.any(valid_mask):
        p_start = max(0, peak_idx - 1)
        p_end = min(len(sub_psd), peak_idx + 2)
        regularity_index = float(np.sum(sub_psd[p_start:p_end]) / (np.sum(sub_psd) + 1e-9))
    else:
        regularity_index = 0.0
        
    feats = {
        "dominant_freq_hz": round(dominant_freq, 2),
        "rms_amplitude": round(rms_amplitude, 4),
        "peak_to_peak_amplitude": round(peak_to_peak, 4),
        "std_amplitude": round(std_amplitude, 4),
        "pd_band_power": round(pd_power, 5),
        "phys_band_power": round(phys_power, 5),
        "pd_power_ratio_pct": round(pd_power_ratio, 1),
        "spectral_entropy": round(spec_entropy, 3),
        "regularity_index": round(regularity_index, 3),
        "sampling_rate_hz": round(fs, 1),
        "source_type": source_label,
        "disclaimer": "Estimated movement features - phone/video sensor estimates may have noise."
    }
    
    psd_data = {
        "frequencies": freqs.tolist(),
        "psd": psd.tolist(),
        "time": time_arr.tolist(),
        "filtered_signal": filtered.tolist(),
        "raw_signal": signal_arr.tolist()
    }
    
    # Generate Matplotlib clinical report figure (dark-theme harmonious)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    
    # Time-domain plot
    ax1.set_facecolor('#1e293b')
    ax1.plot(time_arr, detrended, color="#64748b", alpha=0.6, label="Raw / Detrended", lw=1)
    ax1.plot(time_arr, filtered, color="#38bdf8", label="Filtered Tremor (1.5-15 Hz)", lw=1.6)
    ax1.set_title(f"Tremor Waveform (RMS: {rms_amplitude:.3f}, PtP: {peak_to_peak:.3f})", fontsize=11, fontweight='bold', color='#f8fafc')
    ax1.set_xlabel("Time (seconds)", fontsize=9, color='#94a3b8')
    ax1.set_ylabel("Amplitude (proxy)", fontsize=9, color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(True, linestyle="--", alpha=0.2, color='#475569')
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    ax1.legend(loc="upper right", fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
    
    # Frequency-domain (PSD) plot
    ax2.set_facecolor('#1e293b')
    ax2.plot(freqs, psd, color="#38bdf8", lw=2, label="Power Spectral Density")
    ax2.axvspan(3.5, 7.5, color="#f43f5e", alpha=0.25, label="PD Tremor Band (3.5 - 7.5 Hz)")
    ax2.axvspan(8.0, 12.0, color="#60a5fa", alpha=0.2, label="Physiological Band (8 - 12 Hz)")
    if dominant_freq > 0:
        ax2.axvline(dominant_freq, color="#fb7185", linestyle="--", lw=1.8,
                    label=f"Peak: {dominant_freq:.2f} Hz")
    ax2.set_xlim(0, min(16, fs / 2))
    ax2.set_title(f"Frequency Spectrum (Dominant: {dominant_freq:.2f} Hz | PD Band Power: {pd_power_ratio:.1f}%)",
                  fontsize=11, fontweight='bold', color='#f8fafc')
    ax2.set_xlabel("Frequency (Hz)", fontsize=9, color='#94a3b8')
    ax2.set_ylabel("Power Density", fontsize=9, color='#94a3b8')
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(True, linestyle="--", alpha=0.2, color='#475569')
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    ax2.legend(loc="upper right", fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')
    
    plt.tight_layout()
    return feats, psd_data, fig


def analyze_tremor_video(video_path: str, max_frames: int = 450):
    """
    Extracts involuntary tremor oscillations by tracking hand/finger landmarks in video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
        
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    timestamps = []
    finger_y = []
    annotated_keyframe = None
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, c = frame.shape
        t = frame_count / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            # Track index finger tip (landmark 8) vertical motion relative to wrist (landmark 0)
            y_idx = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
            y_wri = lm[mp_hands.HandLandmark.WRIST].y * h
            rel_y = y_idx - y_wri
            
            timestamps.append(t)
            finger_y.append(rel_y)
            
            if annotated_keyframe is None or frame_count % 30 == 0:
                vis = frame.copy()
                mp_hands_drawing = mp.solutions.drawing_utils
                mp_hands_drawing.draw_landmarks(vis, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.putText(vis, "Tremor Hand Landmark Tracking", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                annotated_keyframe = vis
        frame_count += 1
        
    cap.release()
    hands.close()
    
    if len(finger_y) < 30:
        raise ValueError("Could not detect hand long enough to extract tremor oscillations.")
        
    feats, psd_data, fig = analyze_tremor_signal(np.array(timestamps), np.array(finger_y), source_label="video_landmarks")
    if annotated_keyframe is None:
        annotated_keyframe = np.zeros((400, 600, 3), dtype=np.uint8)
        
    return feats, psd_data, fig, annotated_keyframe


def parse_accelerometer_csv(uploaded_file):
    """
    Parses a user-uploaded CSV file containing phone sensor / accelerometer data.
    Accepts columns: time/timestamp and ax, ay, az (or a single acceleration column).
    """
    df = pd.read_csv(uploaded_file)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    
    # Locate time column
    time_col = None
    for candidate in ['time', 'timestamp', 't', 'seconds', 'sec', 'time(s)']:
        if candidate in cols:
            time_col = candidate
            break
            
    if time_col is None:
        # Generate synthetic time axis assuming 50 Hz
        time_arr = np.linspace(0, len(df) / 50.0, len(df))
    else:
        time_arr = df[time_col].values.astype(float)
        # Normalize start to 0
        time_arr = time_arr - time_arr[0]
        # If timestamp is in milliseconds, convert to seconds
        if len(time_arr) > 1 and (time_arr[-1] - time_arr[0]) > 500:
            time_arr = time_arr / 1000.0
            
    # Locate acceleration axes
    x_col = next((c for c in cols if 'ax' in c or 'acc_x' in c or 'x' == c), None)
    y_col = next((c for c in cols if 'ay' in c or 'acc_y' in c or 'y' == c), None)
    z_col = next((c for c in cols if 'az' in c or 'acc_z' in c or 'z' == c), None)
    
    if x_col and y_col and z_col:
        ax = df[x_col].values.astype(float)
        ay = df[y_col].values.astype(float)
        az = df[z_col].values.astype(float)
        # Magnitude vector
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    elif x_col:
        magnitude = df[x_col].values.astype(float)
    else:
        # Grab first numeric non-time column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            magnitude = df[num_cols[0]].values.astype(float)
        else:
            raise ValueError("No numeric acceleration columns found in CSV.")
            
    return time_arr, magnitude


def generate_synthetic_tremor_data(condition: str = "resting_tremor", duration_sec: float = 10.0, fs: float = 50.0):
    """
    Generates synthetic phone accelerometer data for testing and interactive exploration.
    condition: 'resting_tremor' (4.8 Hz rhythmic oscillation with amplitude modulation),
               'postural_tremor' (6.2 Hz continuous oscillation),
               'physiological' (9.5 Hz low-amplitude benign physiological tremor)
    """
    n = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n)
    
    if condition == "resting_tremor":
        freq = 4.8  # classic Parkinsonian rest tremor frequency (4-6 Hz)
        # Waxing and waning amplitude modulation
        amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
        signal = amp_mod * 0.35 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.04, n)
    elif condition == "postural_tremor":
        freq = 6.2  # postural / action tremor (5-8 Hz)
        signal = 0.22 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.035, n)
    else:
        # Physiological baseline
        freq = 9.8  # normal physiological tremor (8-12 Hz)
        signal = 0.04 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.02, n)
        
    # Add gravity offset (9.81 m/s^2)
    accel_mag = 9.81 + signal
    df = pd.DataFrame({
        "time": np.round(t, 3),
        "accel_mag": np.round(accel_mag, 4)
    })
    return t, accel_mag, df
