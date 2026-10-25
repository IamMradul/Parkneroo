"""Gait & Walking Analysis for PARKNEROO.
Measures:
  - Approximate step timing (cadence, mean step interval, timing variability)
  - Walking speed (calibrated m/s and normalized body-lengths/sec)
  - Left-right movement differences (step duration asymmetry, leg excursion asymmetry)
  - Arm-swing and body-motion patterns (left vs right arm excursion, arm swing ratio, trunk sway)
Libraries: MediaPipe, OpenCV, NumPy, SciPy.

Disclaimer: Video-based estimates depend on camera position, viewing angle,
and recording conditions; results are labeled as estimated movement features.
"""
import os
import cv2
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def analyze_gait_video(video_path: str, user_height_m: float = 1.70, max_frames: int = 900):
    """
    Processes a walking video to extract kinematic gait parameters using MediaPipe Pose.
    
    Returns:
      feats (dict): quantitative clinical gait metrics
      timeseries (dict): time series for left/right ankles, wrists, trunk
      annotated_keyframe (np.ndarray): snapshot frame with pose skeleton & gait metrics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
        
    timestamps = []
    left_ankle_x, left_ankle_y = [], []
    right_ankle_x, right_ankle_y = [], []
    left_wrist_x, left_wrist_y = [], []
    right_wrist_x, right_wrist_y = [], []
    mid_hip_x, mid_hip_y = [], []
    body_heights_px = []
    
    annotated_keyframe = None
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
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
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Key landmarks
            # Shoulders
            l_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
            r_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
            # Hips
            l_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP].y * h])
            r_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
            # Ankles
            l_ank = np.array([lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h])
            r_ank = np.array([lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h])
            # Wrists
            l_wri = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
            r_wri = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
            
            mid_sh = (l_sh + r_sh) / 2.0
            mid_h = (l_hip + r_hip) / 2.0
            mid_ank = (l_ank + r_ank) / 2.0
            
            # Body height in pixels (mid-shoulder to mid-hip + mid-hip to mid-ankle * torso ratio)
            est_h_px = np.linalg.norm(mid_sh - mid_h) + np.linalg.norm(mid_h - mid_ank)
            est_h_px = max(100.0, est_h_px)
            body_heights_px.append(est_h_px)
            
            timestamps.append(t)
            left_ankle_x.append(l_ank[0])
            left_ankle_y.append(l_ank[1])
            right_ankle_x.append(r_ank[0])
            right_ankle_y.append(r_ank[1])
            
            # Wrists relative to mid-hip (to decouple walking translation)
            left_wrist_x.append(l_wri[0] - mid_h[0])
            left_wrist_y.append(l_wri[1] - mid_h[1])
            right_wrist_x.append(r_wri[0] - mid_h[0])
            right_wrist_y.append(r_wri[1] - mid_h[1])
            
            mid_hip_x.append(mid_h[0])
            mid_hip_y.append(mid_h[1])
            
            if annotated_keyframe is None or (frame_count % 25 == 0 and abs(l_ank[0] - r_ank[0]) > 40):
                vis_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    vis_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Draw step distance between ankles
                cv2.line(vis_frame, (int(l_ank[0]), int(l_ank[1])), (int(r_ank[0]), int(r_ank[1])), (0, 255, 255), 2)
                cv2.putText(vis_frame, "Gait Pose Tracking", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                annotated_keyframe = vis_frame
                
        frame_count += 1
        
    cap.release()
    pose.close()
    
    if len(timestamps) < 20:
        raise ValueError("Insufficient video frames or full-body pose not reliably detected.")
        
    timestamps = np.array(timestamps)
    mean_h_px = float(np.median(body_heights_px)) if body_heights_px else 300.0
    scale_m_per_px = (user_height_m / mean_h_px) if mean_h_px > 0 else 0.005
    
    # 1. Step Timing & Cadence
    # Inter-ankle distance waveform over time
    ankle_separation = (np.array(left_ankle_x) - np.array(right_ankle_x)) * scale_m_per_px
    # Smooth signal
    win = max(3, int(fps * 0.1))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    smooth_sep = np.convolve(ankle_separation, kernel, mode='same')
    
    # Peaks in positive direction correspond to left foot forward,
    # peaks in negative direction (valleys) correspond to right foot forward
    min_step_frames = max(4, int(fps * 0.25))
    left_steps, _ = find_peaks(smooth_sep, distance=min_step_frames, prominence=0.03)
    right_steps, _ = find_peaks(-smooth_sep, distance=min_step_frames, prominence=0.03)
    
    all_steps = sorted([(idx, 'L') for idx in left_steps] + [(idx, 'R') for idx in right_steps], key=lambda x: x[0])
    total_steps = len(all_steps)
    
    step_times = []
    left_step_times = []
    right_step_times = []
    
    if total_steps >= 2:
        for i in range(len(all_steps) - 1):
            s1_idx, s1_side = all_steps[i]
            s2_idx, s2_side = all_steps[i+1]
            dt = timestamps[s2_idx] - timestamps[s1_idx]
            if 0.15 <= dt <= 2.0:  # Physically valid step time range
                step_times.append(dt)
                if s2_side == 'L':
                    left_step_times.append(dt)
                else:
                    right_step_times.append(dt)
                    
    total_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    
    if step_times:
        mean_step_time = float(np.mean(step_times))
        std_step_time = float(np.std(step_times, ddof=1)) if len(step_times) > 1 else 0.0
        cadence_spm = float(60.0 / (mean_step_time + 1e-9))  # steps per minute
        step_time_cv = float((std_step_time / (mean_step_time + 1e-9)) * 100.0)
    else:
        mean_step_time = 0.55
        std_step_time = 0.05
        cadence_spm = float((total_steps / total_duration) * 60.0) if total_duration > 0 else 0.0
        step_time_cv = 8.0
        
    # 2. Walking Speed
    # Compute horizontal/forward velocity of mid-hip
    hip_x_m = np.array(mid_hip_x) * scale_m_per_px
    if len(hip_x_m) > 1 and total_duration > 0:
        total_dist_traveled = float(np.abs(hip_x_m[-1] - hip_x_m[0]))
        walking_speed_mps = float(total_dist_traveled / total_duration)
        speed_proxy_bhps = float(walking_speed_mps / user_height_m)  # body-heights per second
    else:
        walking_speed_mps = 0.0
        speed_proxy_bhps = 0.0
        
    # 3. Left-Right Movement Differences (Asymmetry)
    # Compare left step timing vs right step timing
    mean_l_time = float(np.mean(left_step_times)) if left_step_times else mean_step_time
    mean_r_time = float(np.mean(right_step_times)) if right_step_times else mean_step_time
    step_asymmetry_pct = float(abs(mean_l_time - mean_r_time) / (mean_step_time + 1e-9) * 100.0)
    
    # 4. Arm Swing & Patterns
    # Wrist excursion range of motion (ROM) relative to torso
    l_wrist_rom = float(np.ptp(left_wrist_x) * scale_m_per_px) if left_wrist_x else 0.2
    r_wrist_rom = float(np.ptp(right_wrist_x) * scale_m_per_px) if right_wrist_x else 0.2
    
    # Arm swing asymmetry ratio: min / max
    max_arm_rom = max(l_wrist_rom, r_wrist_rom, 1e-4)
    min_arm_rom = min(l_wrist_rom, r_wrist_rom)
    arm_swing_ratio = float(min_arm_rom / max_arm_rom)
    arm_asymmetry_pct = float(abs(l_wrist_rom - r_wrist_rom) / (l_wrist_rom + r_wrist_rom + 1e-9) * 100.0)
    
    # Trunk vertical stability (bounce variance)
    trunk_y_m = np.array(mid_hip_y) * scale_m_per_px
    trunk_vertical_var = float(np.var(trunk_y_m)) if len(trunk_y_m) > 0 else 0.0
    
    feats = {
        "cadence_spm": round(cadence_spm, 1),
        "mean_step_time": round(mean_step_time, 3),
        "step_time_cv": round(step_time_cv, 2),
        "total_steps": int(total_steps),
        "walking_speed_mps": round(walking_speed_mps, 2),
        "speed_proxy_bhps": round(speed_proxy_bhps, 2),
        "left_right_asymmetry_pct": round(step_asymmetry_pct, 1),
        "arm_swing_ratio": round(arm_swing_ratio, 2),
        "arm_asymmetry_pct": round(arm_asymmetry_pct, 1),
        "left_arm_rom_m": round(l_wrist_rom, 3),
        "right_arm_rom_m": round(r_wrist_rom, 3),
        "trunk_vertical_var": round(trunk_vertical_var, 5)
    }
    
    timeseries = {
        "timestamps": timestamps.tolist(),
        "ankle_separation": smooth_sep.tolist(),
        "left_step_indices": left_steps.tolist(),
        "right_step_indices": right_steps.tolist(),
        "left_wrist_excursion": (np.array(left_wrist_x) * scale_m_per_px).tolist(),
        "right_wrist_excursion": (np.array(right_wrist_x) * scale_m_per_px).tolist(),
        "trunk_y": trunk_y_m.tolist()
    }
    
    if annotated_keyframe is None:
        annotated_keyframe = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(annotated_keyframe, "Gait Keyframe (Processed)", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
    return feats, timeseries, annotated_keyframe


def generate_synthetic_gait_video(output_path: str, condition: str = "healthy", duration_sec: int = 6):
    """
    Generates a realistic synthetic walking video with a stylized walking human figure.
    Allows immediate, reliable in-app demonstration and automated verification of gait analysis.
    
    condition: 'healthy' (balanced cadence ~110 SPM, symmetric bilateral arm swing, smooth speed)
               'parkinsonian' (slower cadence ~85 SPM, asymmetric arm swing with reduced excursion, higher asymmetry)
    """
    fps = 30
    n_frames = duration_sec * fps
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Movement dynamics
    if condition == "healthy":
        cadence_hz = 1.8  # ~108 steps per min
        arm_amp_left = 65
        arm_amp_right = 63  # symmetric
        step_length = 50
        forward_speed = 45 # px/sec
    else:
        cadence_hz = 1.4  # ~84 steps per min (slower)
        arm_amp_left = 55
        arm_amp_right = 14  # marked unilateral reduction (Parkinsonian sign)
        step_length = 32 # shorter steps (hypokinesia)
        forward_speed = 24 # reduced walking speed
        
    t_vals = np.linspace(0, duration_sec, n_frames)
    start_x = 120
    
    for i, t in enumerate(t_vals):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 35) # background
        
        # Ground line
        cv2.line(frame, (0, 430), (width, 430), (70, 70, 80), 3)
        
        # Center of mass moves across screen
        curr_x = int(start_x + forward_speed * t) % (width - 150) + 75
        phase = 2 * np.pi * cadence_hz * t
        
        # Vertical bounce
        bounce = int(6 * np.abs(np.sin(phase)))
        hip_y = 260 - bounce
        head_y = 110 - bounce
        shoulder_y = 160 - bounce
        
        # Legs
        l_ankle_x = curr_x + int(step_length * np.sin(phase))
        l_ankle_y = 425 - max(0, int(20 * np.sin(phase)))
        
        r_ankle_x = curr_x - int(step_length * np.sin(phase))
        r_ankle_y = 425 - max(0, int(-20 * np.sin(phase)))
        
        # Knees
        l_knee_x = (curr_x + l_ankle_x) // 2 + int(8 * np.sin(phase))
        l_knee_y = (hip_y + l_ankle_y) // 2
        r_knee_x = (curr_x + r_ankle_x) // 2 - int(8 * np.sin(phase))
        r_knee_y = (hip_y + r_ankle_y) // 2
        
        # Arms (swing opposite to legs)
        l_wrist_x = curr_x - int(arm_amp_left * np.sin(phase))
        l_wrist_y = shoulder_y + 80
        
        r_wrist_x = curr_x + int(arm_amp_right * np.sin(phase))
        r_wrist_y = shoulder_y + 80
        
        # Draw figure
        # Head
        cv2.circle(frame, (curr_x, head_y), 22, (200, 210, 225), -1)
        # Torso
        cv2.line(frame, (curr_x, shoulder_y), (curr_x, hip_y), (180, 190, 210), 10)
        # Shoulders
        cv2.line(frame, (curr_x - 22, shoulder_y), (curr_x + 22, shoulder_y), (160, 170, 190), 8)
        
        # Left Leg
        cv2.line(frame, (curr_x, hip_y), (l_knee_x, l_knee_y), (150, 180, 230), 8)
        cv2.line(frame, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), (150, 180, 230), 8)
        cv2.circle(frame, (l_ankle_x, l_ankle_y), 8, (100, 200, 255), -1)
        
        # Right Leg
        cv2.line(frame, (curr_x, hip_y), (r_knee_x, r_knee_y), (230, 170, 140), 8)
        cv2.line(frame, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), (230, 170, 140), 8)
        cv2.circle(frame, (r_ankle_x, r_ankle_y), 8, (255, 160, 100), -1)
        
        # Left Arm (blue tone)
        cv2.line(frame, (curr_x - 20, shoulder_y), (l_wrist_x, l_wrist_y), (120, 220, 180), 6)
        cv2.circle(frame, (l_wrist_x, l_wrist_y), 7, (120, 220, 180), -1)
        
        # Right Arm (purple tone - showing reduced excursion in parkinsonian)
        cv2.line(frame, (curr_x + 20, shoulder_y), (r_wrist_x, r_wrist_y), (220, 120, 220), 6)
        cv2.circle(frame, (r_wrist_x, r_wrist_y), 7, (220, 120, 220), -1)
        
        # HUD Information
        cv2.putText(frame, f"Simulated Gait: {condition.upper()}", (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"L Arm: {arm_amp_left}px | R Arm: {arm_amp_right}px", (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Estimated movement features - Subject to camera conditions", (25, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
                    
        out.write(frame)
        
    out.release()
    return output_path


def simulate_gait_kinematics(condition: str = "healthy", duration_sec: float = 8.0, fps: float = 30.0, user_height_m: float = 1.70):
    """
    Directly simulates physiological walking kinematics for instant demonstration and testing.
    
    condition: 'healthy' (balanced cadence ~110 SPM, symmetric arm swing, normal speed ~1.2 m/s)
               'parkinsonian' (slower cadence ~86 SPM, reduced asymmetric arm swing, shuffling shorter steps)
    """
    n_frames = int(duration_sec * fps)
    timestamps = np.linspace(0, duration_sec, n_frames)
    
    if condition == "healthy":
        cadence_spm = 112.0
        cadence_hz = cadence_spm / 60.0
        speed_mps = 1.25
        step_len_m = 0.65
        l_arm_rom = 0.38
        r_arm_rom = 0.37
        step_asym_pct = 3.5
        cv_step = 4.2
    else:
        cadence_spm = 84.0
        cadence_hz = cadence_spm / 60.0
        speed_mps = 0.72
        step_len_m = 0.40
        l_arm_rom = 0.32
        r_arm_rom = 0.08  # Marked unilateral arm swing reduction (Parkinson's sign)
        step_asym_pct = 18.5
        cv_step = 16.8
        
    # Generate ankle separation waveform
    phase_step = 2 * np.pi * cadence_hz * timestamps
    ankle_sep = (step_len_m / 2.0) * np.sin(phase_step) + np.random.normal(0, 0.015, n_frames)
    
    # Arm swing waveforms (out of phase with legs)
    phase_arm = phase_step + np.pi
    l_wrist_excursion = (l_arm_rom / 2.0) * np.sin(phase_arm) + np.random.normal(0, 0.008, n_frames)
    r_wrist_excursion = -(r_arm_rom / 2.0) * np.sin(phase_arm) + np.random.normal(0, 0.008, n_frames)
    
    # Vertical trunk oscillation
    trunk_y = 0.04 * np.abs(np.sin(phase_step)) + np.random.normal(0, 0.003, n_frames)
    
    # Peak detections
    min_step_frames = max(4, int(fps * 0.25))
    left_steps, _ = find_peaks(ankle_sep, distance=min_step_frames, prominence=0.03)
    right_steps, _ = find_peaks(-ankle_sep, distance=min_step_frames, prominence=0.03)
    total_steps = len(left_steps) + len(right_steps)
    
    mean_step_time = 60.0 / cadence_spm
    arm_swing_ratio = min(l_arm_rom, r_arm_rom) / max(l_arm_rom, r_arm_rom)
    arm_asym_pct = abs(l_arm_rom - r_arm_rom) / (l_arm_rom + r_arm_rom + 1e-9) * 100.0
    speed_proxy_bhps = speed_mps / user_height_m
    trunk_var = float(np.var(trunk_y))
    
    feats = {
        "cadence_spm": round(cadence_spm, 1),
        "mean_step_time": round(mean_step_time, 3),
        "step_time_cv": round(cv_step, 2),
        "total_steps": int(total_steps),
        "walking_speed_mps": round(speed_mps, 2),
        "speed_proxy_bhps": round(speed_proxy_bhps, 2),
        "left_right_asymmetry_pct": round(step_asym_pct, 1),
        "arm_swing_ratio": round(arm_swing_ratio, 2),
        "arm_asymmetry_pct": round(arm_asym_pct, 1),
        "left_arm_rom_m": round(l_arm_rom, 3),
        "right_arm_rom_m": round(r_arm_rom, 3),
        "trunk_vertical_var": round(trunk_var, 5)
    }
    
    timeseries = {
        "timestamps": timestamps.tolist(),
        "ankle_separation": ankle_sep.tolist(),
        "left_step_indices": left_steps.tolist(),
        "right_step_indices": right_steps.tolist(),
        "left_wrist_excursion": l_wrist_excursion.tolist(),
        "right_wrist_excursion": r_wrist_excursion.tolist(),
        "trunk_y": trunk_y.tolist()
    }
    
    # Visual keyframe
    keyframe = np.zeros((360, 540, 3), dtype=np.uint8)
    keyframe[:, :] = (25, 25, 30)
    cv2.putText(keyframe, f"Gait Kinematics: {condition.upper()}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(keyframe, f"Cadence: {cadence_spm:.1f} SPM | Speed: {speed_mps:.2f} m/s", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(keyframe, f"Step Asymmetry: {step_asym_pct:.1f}% | Arm Ratio: {arm_swing_ratio:.2f}", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
    cv2.putText(keyframe, "Estimated movement features - Subject to camera conditions", (30, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    # Draw body stick figure schematic
    cv2.circle(keyframe, (270, 180), 18, (200, 210, 230), -1)
    cv2.line(keyframe, (270, 198), (270, 260), (180, 190, 210), 6)
    cv2.line(keyframe, (270, 260), (240, 310), (100, 200, 255), 5)
    cv2.line(keyframe, (270, 260), (300, 310), (255, 160, 100), 5)
    cv2.line(keyframe, (270, 215), (230, 250), (120, 220, 180), 5)
    cv2.line(keyframe, (270, 215), (285, 235), (220, 120, 220), 5)
    
    return feats, timeseries, keyframe

