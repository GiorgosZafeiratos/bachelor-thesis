# PPG Signals in the DaLiA Dataset

## Overview
The Daily Life Activities (DaLiA) dataset provides multimodal physiological signals recorded from 18 subjects during everyday activities. This dataset includes PPG, accelerometry, ECG, and EMG signals recorded via wearable devices.

- **PPG**: Photoplethysmography, used to monitor heart rate and blood volume changes.  
- **Sampling frequency**: 64–100 Hz depending on sensor.

---

## Data Format
- **File types**: `.csv` or `.txt`  
- Each row: timestamp + sensor readings  
- Channels: PPG sensors (typically wrist-worn, 1–2 channels)  

Annotations include activity labels (walking, standing, climbing stairs, etc.) and timestamps.

---

## Preprocessing Steps
1. **Signal Parsing**  
   - Load CSV files using `numpy.loadtxt` or `pandas.read_csv`.  

2. **Resampling**  
   - Resample all signals to 64 Hz for uniformity.  

3. **Bandpass Filtering**  
   - Typical range: 0.5–8 Hz to capture heart rate frequency.  
   - Remove baseline drift using high-pass filter and motion artifacts using low-pass filter.

4. **Segmentation**  
   - Sliding windows of 128 samples (~2 s)  
   - 50% overlap  
   - Label each segment according to activity or HR target.

---

## Artefacts
- Motion artifacts from wrist movement  
- Ambient light interference  

Mitigation: smoothing, low-pass filters, or adaptive filtering.

---

## Usage
- Suitable for heart rate estimation, activity classification, or fusion with ECG/EMG data.  

---

## References
1. Reiss A et al. "The DaLiA dataset: A multimodal dataset of daily living activities." PLoS ONE, 2019.  
2. Charlton PH et al. "Photoplethysmography and its applications in heart rate monitoring." Physiol Meas, 2018.
