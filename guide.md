# PENMOT - Complete Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Architecture & How It Works](#architecture--how-it-works)
4. [Setup & Installation](#setup--installation)
5. [Running the Code](#running-the-code)
6. [Testing](#testing)
7. [Output Files](#output-files)
8. [Collaboration Guidelines](#collaboration-guidelines)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

**PENMOT** (Permutation-Equivariant Networks for Multi-Object Tracking) is a geometric deep learning approach to multi-object tracking using Set Transformers and Sinkhorn matching.

### Key Features

- **Set Transformers** for permutation-equivariant feature processing
- **Sinkhorn algorithm** for differentiable matching (replaces Hungarian algorithm)
- **End-to-end training** of the entire pipeline
- **ResNet-50** for appearance feature extraction
- **LSTM + Kalman Filter** for track encoding

### Current Implementation Status

- ✅ Data loading (MOT17 dataset)
- ✅ Detection encoder (ResNet-50)
- ✅ Track encoder (LSTM + Kalman Filter)
- ✅ Kalman Filter for motion prediction
- ✅ Feature visualization tools
- ⏳ Set Transformer (to be implemented)
- ⏳ Sinkhorn matching (to be implemented)
- ⏳ Training loop (to be implemented)
- ⏳ Evaluation metrics (to be implemented)

---

## Folder Structure

```
PENMOT/
├── data/                          # Data loading and preprocessing
│   ├── MOT17/                     # MOT17 dataset (download separately)
│   │   └── MOT17/
│   │       ├── train/             # Training sequences
│   │       │   ├── MOT17-02-FRCNN/
│   │       │   │   ├── img1/      # Frame images
│   │       │   │   ├── gt/        # Ground truth annotations
│   │       │   │   └── det/        # Detection files
│   │       │   └── ...
│   │       └── test/              # Test sequences
│   ├── mot17_dataset.py          # MOT17Dataset class
│   ├── preprocessing.py           # Data preprocessing utilities
│   └── testds.py                 # Dataset testing script
│
├── features/                      # Feature extraction modules
│   ├── detection_encoder.py      # ResNet-50 appearance encoder
│   ├── kalman_filter.py          # Kalman Filter for motion
│   ├── track_encoder.py          # LSTM-based track encoder
│   └── visualizations-features.py # Feature visualization tools
│
├── outputs/                       # All output files
│   └── visualizations/           # Generated visualizations
│       ├── feature_embeddings.png
│       ├── kalman_simulation.png
│       ├── sample_frame.png
│       ├── track_features.png
│       └── objects_per_frame.png
│
├── tests/                         # Test scripts
│   ├── test_dataset.py           # Test dataset loading
│   ├── test_features.py          # Test feature extraction
│   ├── test_kalman.py            # Test Kalman Filter
│   └── test_track_encoder.py     # Test track encoding
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── guide.md                       # This file
└── LICENSE                        # License file
```

### Key Files Explained

**Data Layer:**

- `data/mot17_dataset.py`: Loads MOT17 dataset, returns frames with detections and ground truth
- `data/preprocessing.py`: Preprocesses MOT17 annotations, filters by visibility, converts formats

**Feature Extraction:**

- `features/detection_encoder.py`: Extracts appearance features from bounding boxes using ResNet-50
- `features/kalman_filter.py`: Predicts motion and maintains track state
- `features/track_encoder.py`: Encodes track history using LSTM, combines appearance + motion

**Visualization:**

- `features/visualizations-features.py`: Generates feature analysis plots and visualizations

**Testing:**

- `tests/test_*.py`: Individual test scripts for each component

---

## Architecture & How It Works

### Overall Pipeline

```
Frame t-1 Tracks + Frame t Detections
    ↓
Extract Features (ResNet-50 + Kalman Filter)
    ↓
Track Encoder (LSTM) - Combines appearance + motion
    ↓
[Future: Set Transformer + Sinkhorn Matching]
    ↓
Assignment Matrix → Updated Tracks
```

### Component Details

#### 1. Detection Encoder (`features/detection_encoder.py`)

**Purpose:** Extract appearance features from detection bounding boxes.

**How it works:**

1. Takes an image and bounding boxes as input
2. Crops each bounding box from the image (with 10% padding)
3. Resizes crops to 224×224
4. Passes through ResNet-50 backbone (pretrained on ImageNet)
5. Projects to 512-dimensional feature vector
6. L2 normalizes features

**Key Methods:**

- `forward(img, boxes)`: Extract features for all boxes
- `crop_and_resize(img, boxes)`: Crop and resize boxes

**Example:**

```python
from features.detection_encoder import DetectionEncoder

encoder = DetectionEncoder(feature_dim=512, pretrained=True)
features = encoder(img, boxes)  # [N, 512] features
```

#### 2. Kalman Filter (`features/kalman_filter.py`)

**Purpose:** Predict object motion and maintain track state.

**State Vector:** `[x, y, w, h, vx, vy, vw, vh]`

- `(x, y)`: Center of bounding box
- `(w, h)`: Width and height
- `(vx, vy, vw, vh)`: Velocities

**How it works:**

1. **Initialization:** Creates filter with initial bounding box
2. **Predict:** Predicts next state based on current state and velocity
3. **Update:** Updates state when new observation (detection) arrives
4. **Track Management:** Tracks age, hits, time since last update

**Key Methods:**

- `predict()`: Predict next bounding box
- `update(bbox)`: Update with new observation
- `get_state()`: Get current state vector

**Example:**

```python
from features.kalman_filter import KalmanFilter

kf = KalmanFilter(bbox=[100, 100, 200, 200], track_id=1)
predicted_bbox = kf.predict()  # Predict next position
kf.update(new_bbox)  # Update with observation
```

#### 3. Track Encoder (`features/track_encoder.py`)

**Purpose:** Encode track history using LSTM, combining appearance and motion.

**How it works:**

1. Maintains history of appearance + motion features for each track
2. Uses LSTM to encode temporal sequence
3. Outputs fixed-size track feature vector
4. Combines:
   - **Appearance features** (from DetectionEncoder)
   - **Motion features** (from Kalman Filter state)

**Key Components:**

- `TrackEncoder`: LSTM-based encoder
- `Track`: Individual track with history
- `TrackManager`: Manages multiple tracks

**Key Methods:**

- `TrackEncoder.forward(track_histories)`: Encode track histories
- `TrackManager.init_track(bbox, appearance)`: Initialize new track
- `TrackManager.update_track(track_id, bbox, appearance)`: Update existing track
- `TrackManager.get_track_features()`: Get all track features

**Example:**

```python
from features.track_encoder import TrackEncoder, TrackManager

encoder = TrackEncoder(appearance_dim=512, motion_dim=8, output_dim=512)
manager = TrackManager(encoder)

# Initialize track
track_id = manager.init_track(bbox, appearance_features)

# Update track
manager.update_track(track_id, new_bbox, new_appearance)

# Get all track features
track_features, track_ids = manager.get_track_features()  # [M, 512]
```

#### 4. MOT17 Dataset (`data/mot17_dataset.py`)

**Purpose:** Load MOT17 dataset frames with detections and ground truth.

**How it works:**

1. Loads ground truth annotations from `gt/gt.txt`
2. Filters by visibility threshold (default: 0.25)
3. Converts bounding boxes to `[x1, y1, x2, y2]` format
4. Loads corresponding images
5. Returns batches of frames with detections

**Key Methods:**

- `MOT17Dataset.__getitem__(idx)`: Get frame with detections
- `create_mot17_dataloaders()`: Create train/val dataloaders

**Data Format:**

- `img`: `[3, H, W]` normalized image tensor
- `boxes`: `[N, 4]` bounding boxes `[x1, y1, x2, y2]`
- `ids`: `[N]` object IDs
- `visibility`: `[N]` visibility scores
- `metadata`: Dict with `sequence`, `frame_id`

**Example:**

```python
from data.mot17_dataset import create_mot17_dataloaders

train_loader, val_loader = create_mot17_dataloaders(
    mot17_root="./data/MOT17/MOT17",
    batch_size=1,
    num_workers=0
)

img, boxes, ids, visibility, metadata = next(iter(train_loader))
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works for testing)
- ~10GB disk space for MOT17 dataset

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd PENMOT
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `torch>=2.8.0`: PyTorch
- `torchvision>=0.23.0`: Vision utilities
- `opencv-python>=4.12.0`: Image processing
- `numpy>=2.2.6`: Numerical computing
- `pandas>=2.3.3`: Data handling
- `matplotlib`: Visualization
- `scipy>=1.16.2`: Scientific computing
- `motmetrics>=1.4.0`: MOT evaluation metrics

### Step 3: Download MOT17 Dataset

1. Download from [MOT Challenge](https://motchallenge.net/data/MOT17.zip)
2. Extract to `data/MOT17/`:
   ```bash
   unzip MOT17.zip -d data/
   ```
3. Verify structure:
   ```
   data/MOT17/MOT17/train/MOT17-02-FRCNN/
   ```

### Step 4: Verify Installation

Run a quick test:

```bash
python tests/test_dataset.py
```

If successful, you should see:

- Dataset loaded successfully
- Sample frame information printed

---

## Running the Code

### 1. Test Dataset Loading

```bash
python tests/test_dataset.py
```

**What it does:**

- Loads MOT17 dataset
- Tests data loading pipeline
- Visualizes sample frame with detections
- Saves visualization to `outputs/visualizations/sample_frame.png`

### 2. Test Feature Extraction

```bash
python tests/test_features.py
```

**What it does:**

- Tests DetectionEncoder (ResNet-50)
- Extracts features from sample detections
- Visualizes feature embeddings using PCA
- Saves to `outputs/visualizations/feature_embeddings.png`

### 3. Test Kalman Filter

```bash
python tests/test_kalman.py
```

**What it does:**

- Tests Kalman Filter on simulated track
- Tests on real MOT17 data
- Calculates prediction errors (center error, IoU)
- Visualizes track trajectory
- Saves to `outputs/visualizations/kalman_simulation.png`

**Options:**

```python
# In tests/test_kalman.py, you can adjust:
test_real_mot17_track(max_frames=50)  # Change number of frames
```

### 4. Test Track Encoder

```bash
python tests/test_track_encoder.py
```

**What it does:**

- Tests TrackEncoder (LSTM)
- Tests TrackManager
- Tests on real MOT17 data
- Visualizes track features
- Saves to `outputs/visualizations/track_features.png`

### 5. Generate Feature Visualizations

```bash
python features/visualizations-features.py
```

**What it does:**

- Loads MOT17 data
- Extracts detection and track features
- Generates comprehensive feature analysis:
  - Feature embeddings (PCA)
  - Similarity matrices
  - Intra/inter-class distance analysis
  - Sample tracking visualizations
  - Track history plots
- Saves all to `outputs/visualizations/`

**Output Files:**

- `detection_embeddings.png`: Detection feature PCA
- `track_embeddings.png`: Track feature PCA
- `detection_similarity.png`: Detection similarity matrix
- `track_similarity.png`: Track similarity matrix
- `detection_distances.png`: Intra/inter-class distances
- `track_distances.png`: Track intra/inter-class distances
- `sample_tracking.png`: Sample frame with tracking
- `track_*_history.png`: Individual track histories

### 6. Preprocess Data

```bash
python data/preprocessing.py
```

**What it does:**

- Analyzes MOT17 sequences
- Prints sequence statistics
- Visualizes objects per frame
- Saves to `outputs/visualizations/objects_per_frame.png`

---

## Testing

### Test Structure

Each test file follows this pattern:

1. **Basic Test:** Tests core functionality
2. **Real Data Test:** Tests on actual MOT17 data
3. **Visualization:** Generates plots for inspection

### Running All Tests

```bash
# Test dataset
python tests/test_dataset.py

# Test features
python tests/test_features.py

# Test Kalman Filter
python tests/test_kalman.py

# Test track encoder
python tests/test_track_encoder.py
```

### Expected Outputs

**tests/test_dataset.py:**

- Prints dataset statistics
- Saves `sample_frame.png` with bounding boxes

**tests/test_features.py:**

- Prints feature statistics (mean, std, norm)
- Saves `feature_embeddings.png` with PCA visualization

**tests/test_kalman.py:**

- Prints prediction error statistics (mean, median, max center error, IoU)
- Saves `kalman_simulation.png` with trajectory plot

**tests/test_track_encoder.py:**

- Prints track feature statistics
- Saves `track_features.png` with track feature visualization

---

## Output Files

All outputs are saved to `outputs/visualizations/`:

### Visualization Files

| File                       | Description                       | Generated By                  |
| -------------------------- | --------------------------------- | ----------------------------- |
| `sample_frame.png`         | Sample frame with bounding boxes  | `tests/test_dataset.py`       |
| `feature_embeddings.png`   | Detection feature PCA plot        | `tests/test_features.py`      |
| `kalman_simulation.png`    | Kalman filter trajectory          | `tests/test_kalman.py`        |
| `track_features.png`       | Track feature visualization       | `tests/test_track_encoder.py` |
| `objects_per_frame.png`    | Objects per frame plot            | `data/preprocessing.py`       |
| `detection_embeddings.png` | Detection feature embeddings      | `visualizations-features.py`  |
| `track_embeddings.png`     | Track feature embeddings          | `visualizations-features.py`  |
| `detection_similarity.png` | Detection similarity matrix       | `visualizations-features.py`  |
| `track_similarity.png`     | Track similarity matrix           | `visualizations-features.py`  |
| `detection_distances.png`  | Intra/inter-class distances       | `visualizations-features.py`  |
| `track_distances.png`      | Track intra/inter-class distances | `visualizations-features.py`  |
| `sample_tracking.png`      | Sample tracking visualization     | `visualizations-features.py`  |
| `track_*_history.png`      | Individual track histories        | `visualizations-features.py`  |

### Path Normalization

All paths are normalized using `os.path.normpath(os.path.abspath())` for cross-platform compatibility (Windows/Linux/Mac).

---

## Collaboration Guidelines

### Code Style

1. **Imports:** Group by standard library, third-party, local
2. **Docstrings:** Use triple quotes for all functions/classes
3. **Comments:** Keep only meaningful comments, remove verbose status messages
4. **Path Handling:** Always use `os.path.normpath(os.path.abspath())` for paths
5. **Tensor Operations:** Use `.detach().cpu().numpy()` when converting to NumPy

### File Organization

- **Data:** All data-related code in `data/`
- **Features:** All feature extraction in `features/`
- **Tests:** All test scripts in `tests/` folder
- **Outputs:** All outputs in `outputs/visualizations/`

### Adding New Features

1. **Create module** in appropriate folder (`data/` or `features/`)
2. **Add test script** `tests/test_<feature>.py`
3. **Update this guide** with new functionality
4. **Add visualization** if applicable

### Git Workflow

1. **Branch naming:** `feature/<name>` or `fix/<name>`
2. **Commit messages:** Clear, descriptive
3. **Pull requests:** Include description of changes

### Code Review Checklist

- [ ] Paths normalized for cross-platform compatibility
- [ ] Tensor operations use `.detach()` when needed
- [ ] All outputs saved to `outputs/visualizations/`
- [ ] Test scripts run without errors
- [ ] Docstrings added for new functions
- [ ] No hardcoded paths (use relative paths)

---

## Troubleshooting

### Common Issues

#### 1. MOT17 Dataset Not Found

**Error:** `FileNotFoundError` or "MOT17 dataset not found"

**Solution:**

- Verify dataset is at `data/MOT17/MOT17/`
- Check that `train/` folder exists
- Ensure sequence folders (e.g., `MOT17-02-FRCNN/`) are present

#### 2. Mixed Path Separators (Windows)

**Error:** Paths like `./MOT17/MOT17\\train` with mixed slashes

**Solution:**

- All paths are normalized automatically
- If issue persists, check that `os.path.normpath()` is used after `os.path.join()`

#### 3. Tensor to NumPy Conversion Error

**Error:** `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Solution:**

- Use `.detach().cpu().numpy()` instead of `.cpu().numpy()`
- This detaches the tensor from the computation graph

#### 4. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**

- Reduce batch size in dataloaders
- Use `batch_size=1` for testing
- Set `num_workers=0` to reduce memory usage

#### 5. Import Errors

**Error:** `ModuleNotFoundError` or `ImportError`

**Solution:**

- Ensure you're running from project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python path includes project root

#### 6. Visualization Not Saving

**Error:** Visualizations not appearing in `outputs/visualizations/`

**Solution:**

- Check that `outputs/visualizations/` directory exists
- Verify write permissions
- Check that `os.makedirs(output_dir, exist_ok=True)` is called

### Getting Help

1. **Check logs:** Look for error messages in terminal output
2. **Run tests:** Run individual test scripts to isolate issues
3. **Verify setup:** Ensure MOT17 dataset is correctly placed
4. **Check dependencies:** Verify all packages are installed

---

## Next Steps (Future Implementation)

### To Be Implemented

1. **Set Transformer** (`features/set_transformer.py`)

   - Permutation-equivariant transformer
   - Self-attention on detections and tracks
   - Cross-attention between detections and tracks

2. **Sinkhorn Matching** (`features/sinkhorn.py`)

   - Differentiable matching algorithm
   - Replaces Hungarian algorithm
   - Converts cost matrix to assignment probabilities

3. **PENMOT Model** (`features/penmot.py`)

   - Combines all components
   - End-to-end trainable pipeline

4. **Training Script** (`train.py`)

   - Training loop
   - Loss functions (assignment, contrastive, ID consistency)
   - Checkpointing

5. **Evaluation Script** (`evaluate.py`)

   - MOT metrics (MOTA, IDF1, ID switches)
   - Test set evaluation

6. **Metrics** (`utils/metrics.py`)
   - MOT metric computation
   - Visualization of results

---

## Quick Reference

### Key Imports

```python
# Data
from data.mot17_dataset import MOT17Dataset, create_mot17_dataloaders

# Features
from features.detection_encoder import DetectionEncoder
from features.kalman_filter import KalmanFilter
from features.track_encoder import TrackEncoder, TrackManager

# Visualization
from features.visualizations-features import (
    plot_embeddings,
    plot_similarity_matrix,
    create_feature_analysis_report
)
```

### Common Patterns

```python
# Load dataset
train_loader, val_loader = create_mot17_dataloaders(
    mot17_root="./data/MOT17/MOT17",
    batch_size=1,
    num_workers=0
)

# Extract features
encoder = DetectionEncoder(feature_dim=512, pretrained=True)
features = encoder(img, boxes)

# Track management
manager = TrackManager(track_encoder)
track_id = manager.init_track(bbox, appearance)
manager.update_track(track_id, new_bbox, new_appearance)
```
---
**Last Updated:** 2025-01-27
**Version:** 1.0
