# bachelor-thesis

# Multimodal Biomedical Signal Classification using Hybrid CNN–LSTM with Adaptive Normalization, Attention, and Explainable AI (LRP)

## Abstract
Biomedical signal analysis is crucial in healthcare for monitoring physiological states and diagnosing pathological conditions. This project focuses on multimodal biomedical signal classification leveraging electroencephalography (EEG), electrocardiography (ECG), and electromyography (EMG) data. We propose a hybrid CNN–LSTM architecture enhanced with adaptive normalization, attention mechanisms, and explainable AI (XAI) using Layer-wise Relevance Propagation (LRP).

---

## Architecture Overview
Our framework integrates:

1. **CNN Feature Extractor**:  
   - 3 convolutional blocks with batch normalization and adaptive normalization layers.  
   - Kernel sizes: 3x3 for spatial feature extraction.  
   - Max pooling after each block for dimensionality reduction.

2. **LSTM Temporal Modeling**:  
   - Two stacked LSTM layers to capture temporal dependencies.  
   - Hidden size: 128, with dropout of 0.3.

3. **Attention Mechanism**:  
   - Self-attention module to focus on salient temporal features.  
   - Outputs weighted embeddings for classification.

4. **Classifier**:  
   - Fully connected layers with ReLU activations.  
   - Softmax output for multimodal signal classes.

5. **Explainable AI (LRP)**:  
   - Implements LRP propagation rules.  
   - Provides interpretable signal relevance maps for clinical insights.

---

## Datasets
The framework is evaluated on the following publicly available biomedical datasets:

- **EEG**: PhysioNet EEG Motor Movement/Imagery Dataset  
- **ECG**: MIT-BIH Arrhythmia Dataset  
- **EMG**: NinaPro DB5 Dataset  

Each dataset undergoes preprocessing including filtering, normalization, segmentation, and data augmentation.

---

## Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate biomedical-classifier
