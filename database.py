"""Unified database and longitudinal baseline tracking engine for PARKNEROO.
Connects Voice, Handwriting, Finger Tapping, Gait, and Tremor assessments.
Libraries: SQLite3, Pandas.
"""
import json
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DB_PATH = "sessions.db"

def get_connection(db_path=DB_PATH):
    con = sqlite3.connect(db_path, check_same_thread=False)
    # Ensure all tables exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS voice_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts TEXT NOT NULL,
            prob REAL NOT NULL,
            feats TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS spiral_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts TEXT NOT NULL,
            tremor REAL NOT NULL,
            variance REAL NOT NULL,
            feats TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS finger_tap_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts TEXT NOT NULL,
            tap_count INTEGER NOT NULL,
            mean_iti REAL NOT NULL,
            rhythm_cv REAL NOT NULL,
            amp_decrement REAL NOT NULL,
            tap_speed REAL NOT NULL,
            feats TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS gait_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts TEXT NOT NULL,
            cadence REAL NOT NULL,
            speed_proxy REAL NOT NULL,
            left_right_asymmetry REAL NOT NULL,
            arm_swing_ratio REAL NOT NULL,
            feats TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS tremor_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            ts TEXT NOT NULL,
            peak_freq REAL NOT NULL,
            tremor_power REAL NOT NULL,
            rms_amplitude REAL NOT NULL,
            variability REAL NOT NULL,
            source_type TEXT NOT NULL,
            feats TEXT NOT NULL
        )
    """)
    
    # Backward compatibility with existing 's' and 'spiral_s' tables
    # Migrate any rows from legacy tables if exist and target is empty
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if 's' in tables:
            legacy_voice = con.execute("SELECT user, ts, prob, feats FROM s").fetchall()
            curr_count = con.execute("SELECT COUNT(*) FROM voice_sessions").fetchone()[0]
            if curr_count == 0 and legacy_voice:
                for row in legacy_voice:
                    con.execute("INSERT INTO voice_sessions (user, ts, prob, feats) VALUES (?, ?, ?, ?)", row)
                con.commit()
        if 'spiral_s' in tables:
            legacy_spiral = con.execute("SELECT user, ts, tremor, variance, feats FROM spiral_s").fetchall()
            curr_sp = con.execute("SELECT COUNT(*) FROM spiral_sessions").fetchone()[0]
            if curr_sp == 0 and legacy_spiral:
                for row in legacy_spiral:
                    con.execute("INSERT INTO spiral_sessions (user, ts, tremor, variance, feats) VALUES (?, ?, ?, ?, ?)", row)
                con.commit()
    except Exception:
        pass
        
    return con

def save_voice_session(user: str, prob: float, feats: dict, ts: str = None, con = None):
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    # Also write to legacy 's' if table exists for complete backward compatibility
    try:
        con.execute("CREATE TABLE IF NOT EXISTS s (user TEXT, ts TEXT, prob REAL, feats TEXT)")
        con.execute("INSERT INTO s VALUES (?, ?, ?, ?)", (user, ts, float(prob), json.dumps(feats)))
    except Exception:
        pass
    con.execute(
        "INSERT INTO voice_sessions (user, ts, prob, feats) VALUES (?, ?, ?, ?)",
        (user, ts, float(prob), json.dumps(feats))
    )
    con.commit()
    if close_after:
        con.close()
    return ts

def save_spiral_session(user: str, tremor: float, variance: float, feats: dict, ts: str = None, con = None):
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    try:
        con.execute("CREATE TABLE IF NOT EXISTS spiral_s (user TEXT, ts TEXT, tremor REAL, variance REAL, feats TEXT)")
        con.execute("INSERT INTO spiral_s VALUES (?, ?, ?, ?, ?)", (user, ts, float(tremor), float(variance), json.dumps(feats)))
    except Exception:
        pass
    con.execute(
        "INSERT INTO spiral_sessions (user, ts, tremor, variance, feats) VALUES (?, ?, ?, ?, ?)",
        (user, ts, float(tremor), float(variance), json.dumps(feats))
    )
    con.commit()
    if close_after:
        con.close()
    return ts

def save_finger_tap_session(user: str, tap_count: int, mean_iti: float, rhythm_cv: float,
                            amp_decrement: float, tap_speed: float, feats: dict, ts: str = None, con = None):
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    con.execute(
        """INSERT INTO finger_tap_sessions 
           (user, ts, tap_count, mean_iti, rhythm_cv, amp_decrement, tap_speed, feats) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user, ts, int(tap_count), float(mean_iti), float(rhythm_cv),
         float(amp_decrement), float(tap_speed), json.dumps(feats))
    )
    con.commit()
    if close_after:
        con.close()
    return ts

def save_gait_session(user: str, cadence: float, speed_proxy: float, left_right_asymmetry: float,
                      arm_swing_ratio: float, feats: dict, ts: str = None, con = None):
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    con.execute(
        """INSERT INTO gait_sessions 
           (user, ts, cadence, speed_proxy, left_right_asymmetry, arm_swing_ratio, feats) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user, ts, float(cadence), float(speed_proxy), float(left_right_asymmetry),
         float(arm_swing_ratio), json.dumps(feats))
    )
    con.commit()
    if close_after:
        con.close()
    return ts

def save_tremor_session(user: str, peak_freq: float, tremor_power: float, rms_amplitude: float,
                        variability: float, source_type: str, feats: dict, ts: str = None, con = None):
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    con.execute(
        """INSERT INTO tremor_sessions 
           (user, ts, peak_freq, tremor_power, rms_amplitude, variability, source_type, feats) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user, ts, float(peak_freq), float(tremor_power), float(rms_amplitude),
         float(variability), str(source_type), json.dumps(feats))
    )
    con.commit()
    if close_after:
        con.close()
    return ts

def get_modality_history(user: str, modality: str, con = None) -> pd.DataFrame:
    """Fetch session history for a specific modality and user."""
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    
    table_map = {
        "voice": ("voice_sessions", ["id", "user", "ts", "prob", "feats"]),
        "spiral": ("spiral_sessions", ["id", "user", "ts", "tremor", "variance", "feats"]),
        "finger_tap": ("finger_tap_sessions", ["id", "user", "ts", "tap_count", "mean_iti", "rhythm_cv", "amp_decrement", "tap_speed", "feats"]),
        "gait": ("gait_sessions", ["id", "user", "ts", "cadence", "speed_proxy", "left_right_asymmetry", "arm_swing_ratio", "feats"]),
        "tremor": ("tremor_sessions", ["id", "user", "ts", "peak_freq", "tremor_power", "rms_amplitude", "variability", "source_type", "feats"])
    }
    
    if modality not in table_map:
        raise ValueError(f"Unknown modality: {modality}")
        
    table_name, cols = table_map[modality]
    query = f"SELECT {', '.join(cols)} FROM {table_name} WHERE user=? ORDER BY ts ASC"
    df = pd.read_sql_query(query, con, params=(user,))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        
    if close_after:
        con.close()
    return df

def analyze_baseline_shift(df: pd.DataFrame, key_metric: str, higher_is_worse: bool = True, baseline_n: int = 3):
    """
    Computes baseline from the first baseline_n sessions, calculates deviation of recent sessions,
    and flags persistent changes for review.
    
    Returns a dict with:
      - status: 'Insufficient Data', 'Baseline Building', 'Stable', 'Review Recommended'
      - baseline_mean
      - baseline_std
      - recent_mean (last up to 3 sessions)
      - delta
      - percent_change
      - z_score
      - is_flagged: bool
      - description: str
    """
    if df.empty or len(df) == 0:
        return {
            "status": "No Data",
            "session_count": 0,
            "baseline_mean": None,
            "baseline_std": None,
            "recent_mean": None,
            "delta": 0,
            "percent_change": 0,
            "z_score": 0,
            "is_flagged": False,
            "description": "No sessions recorded yet."
        }
        
    n = len(df)
    if n < baseline_n:
        current_val = df[key_metric].iloc[-1]
        return {
            "status": "Baseline Building",
            "session_count": n,
            "baseline_mean": df[key_metric].mean(),
            "baseline_std": df[key_metric].std(ddof=0) if n > 1 else 0.0,
            "recent_mean": current_val,
            "delta": 0,
            "percent_change": 0,
            "z_score": 0,
            "is_flagged": False,
            "description": f"Establishing personal baseline: {n}/{baseline_n} sessions recorded."
        }
        
    baseline_series = df[key_metric].iloc[:baseline_n]
    base_mean = float(baseline_series.mean())
    base_std = float(baseline_series.std(ddof=0))
    std_floor = max(base_std, 0.05 * abs(base_mean) if abs(base_mean) > 1e-6 else 0.05)
    
    # Recent sessions (up to last 3 sessions after baseline)
    recent_series = df[key_metric].iloc[-3:]
    recent_mean = float(recent_series.mean())
    delta = recent_mean - base_mean
    pct_change = (delta / (abs(base_mean) + 1e-9)) * 100
    z_score = delta / (std_floor + 1e-9)
    
    # Flagging criteria:
    # If higher_is_worse is True, a persistent positive shift (z >= 1.75 or 2 consecutive elevated sessions)
    # If higher_is_worse is False (e.g. speed, arm swing), a persistent drop (z <= -1.75)
    is_elevated_last = (z_score >= 1.75) if higher_is_worse else (z_score <= -1.75)
    
    # Check if last 2 sessions both exceeded threshold
    last_two = df[key_metric].iloc[-2:] if len(df) >= 2 else df[key_metric].iloc[-1:]
    last_two_deviations = [(v - base_mean) / (std_floor + 1e-9) for v in last_two]
    persistent_shift = all((d >= 1.5) if higher_is_worse else (d <= -1.5) for d in last_two_deviations) and len(df) > baseline_n
    
    is_flagged = is_elevated_last or persistent_shift
    
    if is_flagged:
        status = "Review Recommended"
        direction = "increase" if delta > 0 else "decrease"
        desc = (f"Persistent {direction} detected vs your baseline ({base_mean:.2f} -> {recent_mean:.2f}, "
                f"{pct_change:+.1f}%, z={z_score:+.2f}). Consult a healthcare professional.")
    else:
        status = "Stable"
        desc = f"Consistent with your personal baseline ({base_mean:.2f} -> {recent_mean:.2f}, z={z_score:+.2f})."
        
    return {
        "status": status,
        "session_count": n,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "recent_mean": recent_mean,
        "delta": delta,
        "percent_change": pct_change,
        "z_score": z_score,
        "is_flagged": is_flagged,
        "description": desc
    }

def get_all_users(con = None) -> list:
    """Returns a sorted list of unique user IDs across all tables."""
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
    users = set()
    for tbl in ["voice_sessions", "spiral_sessions", "finger_tap_sessions", "gait_sessions", "tremor_sessions"]:
        try:
            res = con.execute(f"SELECT DISTINCT user FROM {tbl}").fetchall()
            for r in res:
                if r[0]:
                    users.add(r[0])
        except Exception:
            pass
    if close_after:
        con.close()
    user_list = sorted(list(users))
    return user_list if user_list else ["demo_user"]

def seed_sample_patient_data(user: str = "demo_user", con = None):
    """
    Populates sample historical data for demonstration and test evaluation
    if the user has fewer than 4 sessions.
    """
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True
        
    # Check if user already has data in finger_tap_sessions
    count = con.execute("SELECT COUNT(*) FROM finger_tap_sessions WHERE user=?", (user,)).fetchone()[0]
    if count >= 3:
        if close_after:
            con.close()
        return False
        
    now = datetime.now()
    # 4 synthetic sessions over the last 3 weeks
    dates = [
        (now - timedelta(days=21)).isoformat(timespec="seconds"),
        (now - timedelta(days=14)).isoformat(timespec="seconds"),
        (now - timedelta(days=7)).isoformat(timespec="seconds"),
        now.isoformat(timespec="seconds")
    ]
    
    # 1. Voice
    voice_scores = [0.18, 0.22, 0.20, 0.38]
    for d, p in zip(dates, voice_scores):
        save_voice_session(user, p, {"MDVP:Fo(Hz)": 154.2, "MDVP:Jitter(%)": 0.005 + p*0.01, "HNR": 24.5 - p*10}, ts=d, con=con)
        
    # 2. Spiral (Handwriting)
    spiral_data = [(1.08, 1.45), (1.11, 1.50), (1.09, 1.48), (1.28, 2.85)]
    for d, (tremor, var) in zip(dates, spiral_data):
        save_spiral_session(user, tremor, var, {"tremor_index": tremor, "thickness_variance": var, "drawing_length": 1420.0}, ts=d, con=con)
        
    # 3. Finger Tapping
    tap_data = [
        (28, 0.35, 12.4, 6.2, 2.8),
        (27, 0.36, 11.8, 5.8, 2.7),
        (28, 0.35, 13.1, 7.0, 2.8),
        (21, 0.48, 24.5, 21.3, 2.1)
    ]
    for d, (cnt, iti, cv, dec, spd) in zip(dates, tap_data):
        save_finger_tap_session(user, cnt, iti, cv, dec, spd, {"tap_count": cnt, "mean_iti": iti, "rhythm_cv": cv, "amp_decrement": dec, "tap_speed": spd}, ts=d, con=con)
        
    # 4. Gait
    gait_data = [
        (108.0, 1.15, 4.2, 0.95),
        (106.0, 1.12, 5.0, 0.93),
        (107.0, 1.14, 4.8, 0.94),
        (94.0, 0.88, 16.5, 0.68)
    ]
    for d, (cad, spd, asym, arm) in zip(dates, gait_data):
        save_gait_session(user, cad, spd, asym, arm, {"cadence": cad, "speed_proxy": spd, "left_right_asymmetry": asym, "arm_swing_ratio": arm}, ts=d, con=con)
        
    # 5. Tremor
    tremor_data = [
        (5.1, 0.045, 0.082, 0.12, "video"),
        (5.0, 0.048, 0.080, 0.11, "video"),
        (5.2, 0.042, 0.079, 0.13, "video"),
        (4.8, 0.145, 0.220, 0.28, "video")
    ]
    for d, (freq, pwr, rms, var, src) in zip(dates, tremor_data):
        save_tremor_session(user, freq, pwr, rms, var, src, {"peak_frequency": freq, "tremor_power": pwr, "rms_amplitude": rms, "movement_variability": var}, ts=d, con=con)
        
    con.commit()
    if close_after:
        con.close()
    return True
