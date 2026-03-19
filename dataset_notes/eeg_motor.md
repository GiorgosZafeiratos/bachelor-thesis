# EEG Motor Movement/Imagery Dataset

## Overview
The EEG Motor Movement/Imagery dataset from PhysioNet contains recordings of 109 subjects performing motor tasks. Each recording captures 64-channel EEG data during:

- Actual motor movements (left/right hand, both hands, both feet)  
- Motor imagery tasks (imagined movement without actual execution)

---

## Data Format
- **File types**: `.edf`  
- **Sampling frequency**: 160 Hz (resampled to 256 Hz in preprocessing)  
- **Channels**: 64 EEG electrodes (10–20 system)  

Each subject performed multiple runs (~3–5 minutes each), with task markers included as events.

---

## Preprocessing Steps
1. **Signal Parsing**  
   - Read `.edf` files using `mne.io.read_raw_edf`.  
   - Extract channels of interest (e.g., 32 channels for standardization).

2. **Resampling**  
   - Resample to 256 Hz to standardize across subjects and runs.  

3. **Bandpass Filtering**  
   - Range: 1–40 Hz  
   - Removes DC offset and high-frequency noise.  

4. **Segmentation**  
   - Sliding windows of 512 samples (~2 s)  
   - 50% overlap  
   - Label each segment based on task annotations.

---

## Artefacts
- Eye blinks, muscle activity, and environmental noise may contaminate signals.  
- Optional preprocessing: ICA or channel rejection for artefact removal.

---

## Usage
- Suitable for classification tasks like left/right-hand movement or motor imagery detection.  
- Supports multimodal fusion with ECG or EMG if available.

---

## References
1. Goldberger AL et al. "EEG Motor Movement/Imagery Dataset." PhysioNet, 2000.  
2. Schalk G et al. "BCI2000: A general-purpose brain-computer interface system." IEEE TNSRE, 2004.
