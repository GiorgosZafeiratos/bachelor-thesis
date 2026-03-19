# MIT-BIH Arrhythmia Dataset

## Overview
The MIT-BIH Arrhythmia Database is one of the most widely used ECG datasets for arrhythmia detection and cardiovascular research. It contains 48 half-hour recordings of 24-hour ambulatory ECGs, sampled at 360 Hz. Each recording includes two channels of ECG data, annotated for beat type and rhythm.

---

## Data Format
- **File types**: `.dat`, `.hea`, `.atr`  
- **Sampling frequency**: 360 Hz  
- **Channels**: 2  
- **Annotations**: Provided in `.atr` files with beat labels following AAMI EC57 standard  

Each recording has a header file containing metadata and signal scaling factors.

---

## Preprocessing Steps
1. **Signal Parsing**  
   Use the `wfdb` Python library to read `.dat` and `.atr` files. Extract the ECG channels required.  

2. **Resampling**  
   - Resample all signals to a consistent 360 Hz (or downsample if needed for uniformity).  
   - Resampling ensures alignment with CNN–LSTM input sizes.

3. **Bandpass Filtering**  
   - Typical range: 0.5–100 Hz  
   - Implemented using Butterworth or FIR filters to remove baseline wander and high-frequency noise.

4. **Segmentation**  
   - Sliding windows of 512 samples (≈1.42 s) with 50% overlap  
   - Each window is labeled according to the majority beat type within the segment.

---

## Signal Quality
- Contains noise from electrode motion and baseline drift.  
- Preprocessing may include denoising with median or wavelet filters.

---

## Usage
- Primarily used for binary or multi-class arrhythmia classification.  
- Can be combined with data augmentation (Gaussian noise, amplitude scaling) to increase robustness.

---

## References
1. Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database." IEEE Eng Med Biol, 2001.  
2. Goldberger AL et al. "PhysioBank, PhysioToolkit, and PhysioNet." Circulation, 2000.  
