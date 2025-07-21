# GACNet: A Graph Attention-Based Neural Network for EEG Localization via Contrastive Learning
GACNet: A Graph Attention-Based Neural Network for EEG Localization via Contrastive Learning
# BiLSTM-GCN Encoder with SupCon for EEG Regression

## Overview

This project proposes a novel deep learning architecture for EEG signal regression, combining spatial and temporal encoding with contrastive learning. The model aims to predict continuous values (e.g., gaze coordinates, emotional arousal, etc.) from multichannel EEG signals.

## Motivation

EEG signals are inherently spatiotemporal, with both channel-wise dependencies and temporal dynamics. Existing models often focus on either spatial or temporal aspects, leading to sub-optimal representations. Furthermore, many regression models lack robust pretraining mechanisms.

## Our Proposal

We propose a hybrid encoder that integrates:
- **BiLSTM**: for capturing temporal dynamics across time points in EEG sequences.
- **GCN**: for modeling spatial relationships between EEG channels using a predefined electrode graph.
- **SupCon Loss**: a supervised contrastive learning objective that encourages the model to learn more discriminative and robust representations during the pretraining stage.

### Architecture

1. **Input**: EEG signal shaped as `(num_channels × num_timepoints)`
2. **GCN Layer**: Learns spatial relationships between EEG electrodes using a graph structure.
3. **BiLSTM Layer**: Encodes temporal features from the spatial embeddings.
4. **Projection Head** (Pretraining Phase): Used for supervised contrastive learning.
5. **Regression Head** (Fine-tuning Phase): Predicts target continuous values.

---

## 🧠 Dataset Format

- EEGEyeNet data should be formatted as tensors of shape:  
  `X: (N_samples, N_channels, N_timepoints)`  
  `Y: (N_samples, 2)` – for (x, y) regression targets.

- For contrastive pretraining, data must be labeled with discrete **class IDs** or **cluster IDs**:
  - `Y_cls: (N_samples,)` where each entry is a class label for SupCon Loss.

- Example:
  ```python
  X.shape = (1000, 129, 500)
  Y.shape = (1000, 2)         # coordinates
  Y_cls.shape = (1000,)       # cluster/class labels for SupCon

  
## ⚙️ Requirements
<pre> ``` 
  python >= 3.8 
  torch >= 1.12 
  torch-geometric >= 2.0 
  numpy 
  scikit-learn 
  matplotlib 
  ``` </pre>



## 🚀 How to Run

## 1. Pretraining with SupCon Loss
```bash

<pre>
```python python train_supcon.py \ --data_path ./data/eeg_dataset.pt \ --output ./checkpoints/encoder.pt \ --temperature 0.07 \ --epochs 100 ``` </pre>

## 2. Fine-tuning for Regression
```bash
python train_regression.py \
    --data_path ./data/eeg_dataset.pt \
    --pretrained ./checkpoints/encoder.pt \
    --output ./checkpoints/regressor.pt \
    --loss mse \
    --epochs 50


