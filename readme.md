# PARKNEROO

It is an AI-powered, non-invasive health monitoring platform designed to help identify and track digital patterns associated with Parkinson’s disease. The system analyses multiple measurable signals such as voice, hand and finger movements, gait, and handwriting using machine learning, signal processing, and computer vision. Instead of relying on a single symptom, PARKNEROO combines information from different modalities and compares changes with an individual’s personal baseline over time. The platform provides an easy-to-understand monitoring report highlighting changes in movement, speech, or other digital biomarkers and encourages timely professional medical evaluation when persistent changes are observed.

**Research prototype. Not a diagnostic tool.**

---

## Central Hub: Personal Baseline & Progress Dashboard
- **This is the feature that connects all your assessments.**
- Saves each session against a unique patient/user ID in SQLite.
- Displays voice, handwriting, finger-tapping, gait, and tremor trends in a unified view.
- Compares current measurements with the user's earlier sessions (personal baseline established from initial sessions).
- Flags persistent multi-session changes ($z \ge 1.75$ or consecutive elevated sessions) for clinical review.
- Multi-metric radar chart and interactive longitudinal timelines with confidence bands.
- **Dynamic Theme Switcher:** Seamlessly toggle between **🌙 Dark Mode** (cyber-clinical glassmorphic design) and **☀️ Light Mode** (crisp, high-contrast medical white) with adaptive Plotly charts and customized typography.
- **Libraries:** Pandas, Plotly, SQLite.
- **Strict Clinical Rule:** *Never interpret a change by itself as proof of Parkinson’s disease.*

---

## Modalities & Clinical Features

### 1. Voice Analysis
- **Symptom-related changes:** Vocal tremor, hoarseness, dysphonia, reduced pitch variation.
- **User does:** Uploads a 3-5 second audio recording of sustained phonation ('aaah').
- **Measurements:** Fundamental frequency ($F_0$), Jitter, Shimmer, Harmonics-to-Noise Ratio (HNR), and Parkinson's probability with SHAP waterfall explainability.
- **Libraries:** Praat (Parselmouth), Scikit-Learn, XGBoost, SHAP.

### 2. Handwriting & Movement (Spiral Drawing)
- **Symptom-related changes:** Micrographia, pen pressure hesitations, tremor on continuous curves.
- **User does:** Uploads a spiral drawing image.
- **Measurements:** Tremor Index (arc length divergence), Thickness Variance (proxy for pressure hesitation), and Drawing Length.
- **Libraries:** OpenCV, NumPy.

### 3. Finger-Tapping Test
- **Symptom-related changes:** Slowness of movement (bradykinesia), reduced movement amplitude (hypometria / fatigue), and difficulty maintaining a rhythm (arrhythmokinesis).
- **User does:** Repeatedly taps thumb and index finger in front of the camera with widest possible excursion.
- **Code measures:**
  - Number of taps
  - Time between taps (Inter-Tap Interval / ITI in seconds: mean and median)
  - Tap rhythm variability (Coefficient of Variation % of ITI)
  - Change in finger movement amplitude (amplitude decrement % & regression slope)
  - Tapping speed (Hz)
- **Libraries:** MediaPipe Hands, OpenCV, NumPy, SciPy.

### 4. Gait & Walking Analysis
- **Symptom-related changes:** Shorter steps (hypokinesia), slower walking, irregular timing, and altered / diminished arm swing (often asymmetrical).
- **User does:** Records a short walking video in a suitable space.
- **Code measures:**
  - Approximate step timing and cadence (steps per minute)
  - Walking speed (calibrated m/s based on user height) and speed proxy (body-heights/sec)
  - Left–right movement differences (step duration asymmetry %)
  - Arm-swing and body-motion patterns (bilateral arm range of motion, arm swing ratio, trunk vertical bounce variance)
- **Libraries:** MediaPipe Pose, OpenCV, NumPy, SciPy.
- **Notice:** *Video-based estimates depend on camera position, viewing angle, and recording conditions. Results are labeled as estimated movement features.*

### 5. Tremor & Movement Signal Analysis
- **Symptom-related changes:** Rhythmic shaking or other involuntary movement patterns (characteristic resting tremor: 4–6 Hz, postural tremor: 5–8 Hz).
- **User does:** Performs a standardized hand-position task (outstretched hand) or completes a guided movement test, recorded via phone accelerometer sensor (CSV) or video.
- **Code measures:**
  - Signal amplitude (Peak-to-Peak amplitude, Root Mean Square / RMS acceleration/displacement)
  - Frequency components (FFT, Welch Power Spectral Density, dominant peak frequency in Hz, power in 3.5–7.5 Hz Parkinsonian band vs 8–12 Hz Physiological band)
  - Movement variability (Spectral entropy, tremor regularity index)
  - Changes across repeated sessions
- **Libraries:** SciPy, NumPy, Matplotlib, Plotly, MediaPipe Hands.
- **Notice:** *A phone sensor or video does not capture every tremor reliably; label results as estimated movement features.*

---

## How to Run

### Activate Conda Environment
```bash
conda activate parkneroo
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch Platform
```bash
streamlit run app.py
```

---

## Architecture & File Structure
```
Parknero/
├── app.py                   # Unified Streamlit application & interactive dashboard
├── database.py              # SQLite storage, longitudinal baseline & shift detection engine
├── features.py              # Praat-based acoustic feature extraction
├── vision_features.py       # OpenCV spiral drawing kinematic extraction
├── finger_tap_analysis.py   # MediaPipe hand tracking, tap rhythm, and amplitude fatigue
├── gait_analysis.py         # MediaPipe pose tracking, cadence, speed, and arm swing ratio
├── tremor_analysis.py       # SciPy signal processing, Welch PSD, FFT, and amplitude metrics
├── train.py                 # UCI dataset subject-grouped cross-validation trainer
├── sessions.db              # SQLite database storing multi-modal sessions
├── requirements.txt         # Pinned project dependencies
└── readme.md                # Project documentation
```

---

## Known Limitations & Ethical Guidelines
- **Not a Diagnostic Instrument:** Intended solely as a research and personal longitudinal monitoring tool.
- **Biological & Behavioral Variability:** Movement and speech changes can result from normal fatigue, circadian variation, emotional state, caffeine, or minor illnesses.
- **Sensor & Environmental Factors:** Lighting, camera frame rates, smartphone accelerometer calibration, and microphone distance introduce noise that should not be mistaken for clinical progression.
- **Clinical Review:** Any persistent baseline shift should always be assessed in person by a qualified neurologist.