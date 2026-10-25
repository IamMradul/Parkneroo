"""Finger-Tapping Test Analysis for PARKNEROO.
Measures:
  - Number of taps
  - Time between taps (Inter-Tap Interval / ITI)
  - Tap rhythm variability (CV of ITI)
  - Change in finger movement amplitude (amplitude decrement / fatigue slope)
  - Tapping speed (Hz)
Libraries: MediaPipe, OpenCV, Python, NumPy, SciPy.
"""
import os
import cv2
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def analyze_finger_tapping_video(video_path: str, max_frames: int = 900):
    """
    Analyzes a video of a user performing the finger-tapping test
    (thumb and index finger tapping repeatedly).
    
    Returns:
      feats (dict): quantitative clinical metrics
      timeseries (dict): time, distance, tap indices for plotting
      annotated_keyframe (np.ndarray): snapshot frame with landmarks & overlays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback standard fps
        
    distances = []
    timestamps = []
    frame_indices = []
    keyframe = None
    annotated_keyframe = None
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, c = frame.shape
        t = frame_count / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Landmark 4: Thumb Tip, Landmark 8: Index Tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            # Calculate pixel coordinates
            p_thumb = np.array([thumb_tip.x * w, thumb_tip.y * h])
            p_index = np.array([index_tip.x * w, index_tip.y * h])
            p_wrist = np.array([wrist.x * w, wrist.y * h])
            p_mcp = np.array([mcp_middle.x * w, mcp_middle.y * h])
            
            # Palm scale normalization factor to eliminate distance-from-camera artifact
            palm_scale = np.linalg.norm(p_wrist - p_mcp)
            if palm_scale < 10.0:
                palm_scale = 100.0
                
            dist_norm = np.linalg.norm(p_thumb - p_index) / palm_scale
            
            distances.append(dist_norm)
            timestamps.append(t)
            frame_indices.append(frame_count)
            
            # Keep one clear frame with annotations as preview keyframe
            if annotated_keyframe is None or (frame_count % 30 == 0 and dist_norm > 0.4):
                vis_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    vis_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # Draw aperture line between thumb and index
                pt1 = (int(p_thumb[0]), int(p_thumb[1]))
                pt2 = (int(p_index[0]), int(p_index[1]))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 3)
                cv2.putText(vis_frame, f"Aperture: {dist_norm:.2f}", (pt1[0], max(30, pt1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                annotated_keyframe = vis_frame
        else:
            # If hand not detected in this frame, carry forward or mark missing
            if distances:
                distances.append(distances[-1])
            else:
                distances.append(0.0)
            timestamps.append(t)
            frame_indices.append(frame_count)
            
        frame_count += 1
        
    cap.release()
    hands.close()
    
    if len(distances) < 15:
        raise ValueError("Insufficient video frames or no hand was reliably detected.")
        
    distances = np.array(distances)
    timestamps = np.array(timestamps)
    
    # Smooth signal to suppress high-frequency jitter
    window_len = max(3, int(fps * 0.08))
    if window_len % 2 == 0:
        window_len += 1
    kernel = np.ones(window_len) / window_len
    smoothed_dist = np.convolve(distances, kernel, mode='same')
    
    # Peak and Valley detection:
    # Valleys (inverted peaks) correspond to tap closures (fingers touching)
    # Distance between consecutive valleys gives the Inter-Tap Interval (ITI)
    # Peaks between valleys represent the maximum opening amplitude of that tap cycle
    inv_signal = -smoothed_dist
    min_dist_frames = max(3, int(fps * 0.15)) # Taps can't physically exceed ~7 taps/sec
    valleys, _ = find_peaks(inv_signal, distance=min_dist_frames, prominence=0.04)
    peaks, _ = find_peaks(smoothed_dist, distance=min_dist_frames, prominence=0.04)
    
    tap_count = len(valleys)
    
    if tap_count >= 2:
        tap_times = timestamps[valleys]
        itis = np.diff(tap_times)  # Inter-Tap Intervals in seconds
        
        mean_iti = float(np.mean(itis))
        median_iti = float(np.median(itis))
        std_iti = float(np.std(itis, ddof=1)) if len(itis) > 1 else 0.0
        rhythm_cv = float((std_iti / (mean_iti + 1e-9)) * 100.0)  # % Coefficient of Variation
        tap_speed = float(1.0 / (mean_iti + 1e-9))  # Taps per second (Hz)
        
        # Calculate opening amplitude between each tap cycle
        cycle_amplitudes = []
        for i in range(len(valleys) - 1):
            v_start = valleys[i]
            v_end = valleys[i+1]
            cycle_segment = smoothed_dist[v_start:v_end]
            if len(cycle_segment) > 0:
                cycle_amplitudes.append(float(np.max(cycle_segment) - np.min(cycle_segment)))
                
        if cycle_amplitudes:
            k = len(cycle_amplitudes)
            half = max(1, k // 2)
            first_half_amp = float(np.mean(cycle_amplitudes[:half]))
            second_half_amp = float(np.mean(cycle_amplitudes[half:]))
            # Amplitude decrement percentage: positive means amplitude reduced over time (fatigue/hypometria)
            amp_decrement = float(((first_half_amp - second_half_amp) / (first_half_amp + 1e-9)) * 100.0)
            
            # Amplitude slope
            x_cycles = np.arange(k)
            if k > 1:
                slope, _ = np.polyfit(x_cycles, cycle_amplitudes, 1)
            else:
                slope = 0.0
            amp_slope = float(slope)
        else:
            amp_decrement = 0.0
            amp_slope = 0.0
            cycle_amplitudes = [0.0]
    else:
        # Fallback if very few taps detected
        mean_iti = 0.0
        median_iti = 0.0
        std_iti = 0.0
        rhythm_cv = 0.0
        tap_speed = 0.0
        amp_decrement = 0.0
        amp_slope = 0.0
        cycle_amplitudes = []
        itis = np.array([])
        
    total_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    
    feats = {
        "tap_count": int(tap_count),
        "mean_iti": round(mean_iti, 3),
        "median_iti": round(median_iti, 3),
        "std_iti": round(std_iti, 3),
        "rhythm_cv": round(rhythm_cv, 2),
        "amp_decrement": round(amp_decrement, 2),
        "amp_slope": round(amp_slope, 4),
        "tap_speed": round(tap_speed, 2),
        "total_duration": round(total_duration, 2),
        "mean_amplitude": round(float(np.mean(cycle_amplitudes)) if cycle_amplitudes else 0.0, 3)
    }
    
    timeseries = {
        "timestamps": timestamps.tolist(),
        "distances": distances.tolist(),
        "smoothed_dist": smoothed_dist.tolist(),
        "tap_indices": valleys.tolist(),
        "peak_indices": peaks.tolist(),
        "itis": itis.tolist(),
        "cycle_amplitudes": cycle_amplitudes
    }
    
    if annotated_keyframe is None:
        # Default placeholder keyframe if no hands visible
        annotated_keyframe = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(annotated_keyframe, "Hand Keyframe (Processed)", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
    return feats, timeseries, annotated_keyframe


def simulate_finger_tapping_kinematics(condition: str = "healthy", duration_sec: float = 8.0, fps: float = 30.0):
    """
    Directly simulates physiological finger-tapping kinematic waveforms and extracts clinical metrics.
    Ideal for instant interactive dashboard demonstration and baseline benchmarking.
    
    condition: 'healthy' (steady rhythm ~3.2 Hz, preserved amplitude)
               'parkinsonian' (bradykinesia, arrhythmokinesis / high CV, amplitude decrement / fatigue)
    """
    n_frames = int(duration_sec * fps)
    timestamps = np.linspace(0, duration_sec, n_frames)
    
    if condition == "healthy":
        base_freq = 3.2
        cv_noise = 0.04
        amp_decay_rate = 0.03
        base_amp = 0.85
    else:
        # Parkinsonian: slower, erratic intervals, progressive amplitude reduction (hypometria)
        base_freq = 2.1
        cv_noise = 0.24
        amp_decay_rate = 0.38
        base_amp = 0.75
        
    distances = []
    current_phase = 0.0
    
    for i, t in enumerate(timestamps):
        # Time-varying instantaneous frequency (slowing down if PD)
        inst_freq = base_freq * (1.0 - 0.15 * (t / duration_sec)) if condition != "healthy" else base_freq
        inst_freq += np.random.normal(0, cv_noise)
        current_phase += 2 * np.pi * max(0.5, inst_freq) * (1.0 / fps)
        
        # Amplitude envelope decay
        amp_envelope = base_amp * max(0.2, 1.0 - (amp_decay_rate * (t / duration_sec)))
        
        # Aperture distance: 0 (closed) to amp_envelope (wide open)
        d = amp_envelope * (0.5 + 0.5 * np.cos(current_phase)) + np.random.normal(0, 0.015)
        d = max(0.02, d)
        distances.append(d)
        
    distances = np.array(distances)
    
    # Peak and valley detection
    inv_signal = -distances
    min_dist_frames = max(3, int(fps * 0.15))
    valleys, _ = find_peaks(inv_signal, distance=min_dist_frames, prominence=0.04)
    peaks, _ = find_peaks(distances, distance=min_dist_frames, prominence=0.04)
    
    tap_count = len(valleys)
    tap_times = timestamps[valleys]
    itis = np.diff(tap_times) if len(tap_times) > 1 else np.array([])
    
    if len(itis) > 0:
        mean_iti = float(np.mean(itis))
        median_iti = float(np.median(itis))
        std_iti = float(np.std(itis, ddof=1)) if len(itis) > 1 else 0.0
        rhythm_cv = float((std_iti / (mean_iti + 1e-9)) * 100.0)
        tap_speed = float(1.0 / (mean_iti + 1e-9))
        
        cycle_amplitudes = []
        for i in range(len(valleys) - 1):
            v_start = valleys[i]
            v_end = valleys[i+1]
            cycle_segment = distances[v_start:v_end]
            if len(cycle_segment) > 0:
                cycle_amplitudes.append(float(np.max(cycle_segment) - np.min(cycle_segment)))
                
        if cycle_amplitudes:
            k = len(cycle_amplitudes)
            half = max(1, k // 2)
            first_half_amp = float(np.mean(cycle_amplitudes[:half]))
            second_half_amp = float(np.mean(cycle_amplitudes[half:]))
            amp_decrement = float(((first_half_amp - second_half_amp) / (first_half_amp + 1e-9)) * 100.0)
            x_cycles = np.arange(k)
            amp_slope = float(np.polyfit(x_cycles, cycle_amplitudes, 1)[0]) if k > 1 else 0.0
        else:
            amp_decrement = 0.0
            amp_slope = 0.0
            cycle_amplitudes = [0.0]
    else:
        mean_iti = 0.0
        median_iti = 0.0
        std_iti = 0.0
        rhythm_cv = 0.0
        tap_speed = 0.0
        amp_decrement = 0.0
        amp_slope = 0.0
        cycle_amplitudes = []
        
    feats = {
        "tap_count": int(tap_count),
        "mean_iti": round(mean_iti, 3),
        "median_iti": round(median_iti, 3),
        "std_iti": round(std_iti, 3),
        "rhythm_cv": round(rhythm_cv, 2),
        "amp_decrement": round(amp_decrement, 2),
        "amp_slope": round(amp_slope, 4),
        "tap_speed": round(tap_speed, 2),
        "total_duration": round(float(duration_sec), 2),
        "mean_amplitude": round(float(np.mean(cycle_amplitudes)) if cycle_amplitudes else 0.0, 3)
    }
    
    timeseries = {
        "timestamps": timestamps.tolist(),
        "distances": distances.tolist(),
        "smoothed_dist": distances.tolist(),
        "tap_indices": valleys.tolist(),
        "peak_indices": peaks.tolist(),
        "itis": itis.tolist(),
        "cycle_amplitudes": cycle_amplitudes
    }
    
    # Generate visualization keyframe
    keyframe = np.zeros((360, 540, 3), dtype=np.uint8)
    keyframe[:, :] = (25, 25, 30)
    cv2.putText(keyframe, f"Finger Tapping: {condition.upper()}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)
    cv2.putText(keyframe, f"Taps Detected: {tap_count} | Speed: {tap_speed:.1f} Hz", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(keyframe, f"Rhythm CV: {rhythm_cv:.1f}% | Decrement: {amp_decrement:.1f}%", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
    # Draw hand landmark schematic
    cv2.circle(keyframe, (270, 260), 45, (100, 110, 130), -1)
    cv2.line(keyframe, (230, 240), (190, 190), (140, 160, 200), 10)
    cv2.circle(keyframe, (190, 190), 14, (0, 255, 255), -1)
    cv2.line(keyframe, (270, 220), (270, 160), (140, 160, 200), 10)
    cv2.circle(keyframe, (270, 160), 14, (0, 255, 255), -1)
    cv2.line(keyframe, (190, 190), (270, 160), (0, 255, 0), 3)
    cv2.putText(keyframe, "Normalized Aperture Tracking", (30, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                
    return feats, timeseries, keyframe

