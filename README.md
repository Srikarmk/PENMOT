# PENMOT - Permutation-Equivariant Networks for Multi-Object Tracking

A geometric deep learning approach to multi-object tracking using Set Transformers and Sinkhorn matching.

## What is this?

PENMOT replaces the traditional Hungarian algorithm in object tracking with a learned, permutation-equivariant network. Instead of hand-crafted matching, we use:
- **Set Transformers** for permutation-equivariant feature processing
- **Sinkhorn algorithm** for differentiable matching
- **End-to-end training** of the entire pipeline

## Quick Start

### Install Dependencies
```bash
pip install torch torchvision opencv-python scipy numpy motmetrics
```

### Download Dataset
```bash
# Download MOT17 from https://motchallenge.net/data/MOT17.zip
# Extract to data/MOT17/
```

### Train
```bash
python train.py --data_root data/MOT17 --epochs 50
```

### Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Project Breakdown

### 1. Data & Features
- Load MOT17 dataset (detections + ground truth tracks)
- Extract appearance features using ResNet-50
- Build motion features using Kalman filter
- Create ground truth assignment matrices

**Files to implement:**
- `utils/data_loader.py` - Dataset loading
- `models/detection_encoder.py` - ResNet-50 feature extractor
- `models/track_encoder.py` - LSTM + Kalman filter for tracks

### 2. Model Architecture
- Set Transformer (self-attention + cross-attention)
- Sinkhorn matching algorithm
- Complete PENMOT model

**Files to implement:**
- `models/set_transformer.py` - Permutation-equivariant transformer
- `models/sinkhorn.py` - Differentiable matching
- `models/penmot.py` - Main model combining all components

### 3. Training & Evaluation
- Loss functions (assignment, contrastive, ID consistency)
- Training loop with logging
- MOT metrics (MOTA, IDF1, ID switches)
- Ablation studies

**Files to implement:**
- `train.py` - Training script
- `evaluate.py` - Evaluation on test set
- `utils/metrics.py` - MOT metric computation

## How It Works

### Pipeline
```
Frame t-1 Tracks + Frame t Detections
    ↓
Extract Features (ResNet + Kalman)
    ↓
Set Transformer (no positional encoding)
    ↓
Compute Affinity Matrix
    ↓
Sinkhorn Matching
    ↓
Assignment Matrix → Updated Tracks
```

### Key Components

**Set Transformer**
- Processes detections and tracks as unordered sets
- Self-attention on detections, self-attention on tracks
- Cross-attention between detections and tracks
- No positional encodings = permutation equivariant

**Sinkhorn Algorithm**
- Converts cost matrix to assignment probabilities
- Fully differentiable (unlike Hungarian)
- Iteratively normalizes rows and columns

**Training Losses**
- Assignment loss: Match predicted to ground truth assignments
- Contrastive loss: Similar features for matched pairs
- ID consistency: Penalize identity switches

## Project Structure

```
PENMOT/
├── data/
│   └── MOT17/          # Dataset goes here
├── models/
│   ├── detection_encoder.py
│   ├── track_encoder.py
│   ├── set_transformer.py
│   ├── sinkhorn.py
│   └── penmot.py
├── utils/
│   ├── data_loader.py
│   ├── kalman_filter.py
│   └── metrics.py
├── train.py
├── evaluate.py
└── README.md
```

## Implementation Notes

### Week-by-Week Breakdown

**Week 1-2:** Data pipeline + feature extraction
- MOT17 data loader
- ResNet-50 for appearance
- Kalman filter for motion

**Week 3:** Core architecture
- Set Transformer implementation
- Sinkhorn algorithm

**Week 4:** Training
- Loss functions
- Training loop
- Hyperparameter tuning

**Week 5:** Experiments
- Run on MOT17 test set
- Ablations (equivariance, Sinkhorn vs Hungarian, scene density)

**Week 6:** Final evaluation + results

## Expected Performance

- **Target:** MOTA > 75%, IDF1 > 70%
- **Comparison:** ByteTrack baseline (MOTA 80.3%)
- **Focus:** Better performance in crowded scenes, fewer ID switches

## Notes

- Use pre-trained ResNet-50 (don't train from scratch)
- Limit max detections to 100 per frame for efficiency
- Test permutation equivariance with unit tests
- Visualize attention maps to debug

## References

- ByteTrack (ECCV 2022) - baseline method
- Set Transformer (ICML 2019) - architecture
- Sinkhorn Distances (NeurIPS 2013) - matching algorithm
