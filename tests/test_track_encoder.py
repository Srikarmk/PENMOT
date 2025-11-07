import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import torch (optional - only needed for some tests)
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch not available ({e}). Some tests will be skipped.")

# Try to import MOT17 dataset (optional - only needed for test 3)
try:
    from data.mot17_dataset import create_mot17_dataloaders
    MOT17_AVAILABLE = True and TORCH_AVAILABLE
except (ImportError, OSError) as e:
    MOT17_AVAILABLE = False
    if TORCH_AVAILABLE:
        print(f"Warning: MOT17 dataset not available ({e}). Test 3 will be skipped.")

# Try to import detection encoder (optional - only needed for test 3)
try:
    from features.detection_encoder import DetectionEncoder
    DETECTION_ENCODER_AVAILABLE = True and TORCH_AVAILABLE
except (ImportError, OSError) as e:
    DETECTION_ENCODER_AVAILABLE = False
    if TORCH_AVAILABLE:
        print(f"Warning: DetectionEncoder not available ({e}). Test 3 will be skipped.")

# Try to import track encoder (required for tests 1 and 2)
try:
    from features.track_encoder import TrackEncoder, Track, TrackManager
    TRACK_ENCODER_AVAILABLE = True and TORCH_AVAILABLE
except (ImportError, OSError) as e:
    TRACK_ENCODER_AVAILABLE = False
    print(f"Error: TrackEncoder not available ({e}). Basic tests cannot run.")
    if not TORCH_AVAILABLE:
        print("  This is likely due to PyTorch DLL issues.")


def test_basic_track_encoder():
    """Test basic TrackEncoder functionality."""
    print("TEST 1: Basic Track Encoder")
    
    if not TRACK_ENCODER_AVAILABLE:
        print("Skipping test - TrackEncoder not available")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder = TrackEncoder(
        appearance_dim=512,
        motion_dim=8,
        hidden_dim=256,
        output_dim=512,
        track_history_len=10
    ).to(device)
    
    track_history = []
    for t in range(15):
        appearance = torch.randn(512)
        motion = torch.randn(8)
        track_history.append({
            'appearance': appearance,
            'motion': motion
        })
    
    print(f"Created track history: {len(track_history)} frames")
    
    with torch.no_grad():
        features = encoder([track_history])
    
    print(f"Encoded features shape: {features.shape} (expected: [1, 512])")
    print(f"Feature norm: {features.norm(dim=1).item():.3f}")
    
    return encoder


def test_multiple_tracks():
    """Test encoding multiple tracks."""
    print("\nTEST 2: Multiple Tracks")
    
    if not TRACK_ENCODER_AVAILABLE:
        print("Skipping test - TrackEncoder not available")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TrackEncoder().to(device)
    
    histories = []
    for track_id in range(5):
        history_len = np.random.randint(5, 20)
        history = []
        for t in range(history_len):
            appearance = torch.randn(512)
            motion = torch.randn(8)
            history.append({
                'appearance': appearance,
                'motion': motion
            })
        histories.append(history)
        print(f"Track {track_id}: {len(history)} frames")
    
    with torch.no_grad():
        features = encoder(histories)
    
    print(f"Encoded {len(histories)} tracks")
    print(f"Features shape: {features.shape} (expected: [{len(histories)}, 512])")
    
    similarity_matrix = features @ features.T
    off_diagonal = similarity_matrix - torch.eye(len(histories), device=device)
    avg_similarity = off_diagonal.abs().mean().item()
    
    print(f"Average inter-track similarity: {avg_similarity:.3f}")
    
    return encoder


def test_real_mot17_tracks():
    """Test on real MOT17 data."""
    print("\nTEST 3: Real MOT17 Tracks")
    
    if not (MOT17_AVAILABLE and DETECTION_ENCODER_AVAILABLE and TRACK_ENCODER_AVAILABLE):
        print("Skipping MOT17 test - required modules not available")
        return None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    detection_encoder = DetectionEncoder(feature_dim=512, pretrained=True).to(device)
    detection_encoder.eval()
    
    track_encoder = TrackEncoder().to(device)
    track_encoder.eval()
    
    track_manager = TrackManager(track_encoder)
    
    possible_paths = [
        os.path.join(project_root, "data", "MOT17", "MOT17"),
        os.path.join(project_root, "MOT17", "MOT17"),
        os.path.join(project_root, "data", "MOT17", "MOT17"),
    ]
    
    MOT17_PATH = None
    for path in possible_paths:
        normalized_path = os.path.normpath(os.path.abspath(path))
        if os.path.exists(normalized_path):
            train_path = os.path.join(normalized_path, 'train')
            train_path = os.path.normpath(train_path)
            if os.path.exists(train_path):
                MOT17_PATH = normalized_path
                break
    
    if MOT17_PATH is None:
        print(f"Skipping MOT17 test - dataset not found")
        print(f"Checked paths: {possible_paths}")
        return None, None, None
    
    print(f"Using MOT17 dataset at: {MOT17_PATH}")
    
    try:
        train_loader, _ = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)
    except (FileNotFoundError, OSError, Exception) as e:
        print(f"Skipping MOT17 test - error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    print("\nProcessing frames...")
    frame_count = 0
    max_frames = 30
    current_sequence = None
    
    for img, boxes, ids, visibility, metadata in train_loader:
        if frame_count >= max_frames:
            break
        
        if current_sequence is None:
            current_sequence = metadata['sequence']
        elif metadata['sequence'] != current_sequence:
            continue
        
        img = img.to(device)
        boxes = boxes.to(device)
        
        if len(boxes) == 0:
            frame_count += 1
            continue
        
        with torch.no_grad():
            appearances = detection_encoder(img, boxes)
        
        for i in range(len(boxes)):
            box = boxes[i]
            obj_id = ids[i].item()
            appearance = appearances[i]
            
            box_np = box.detach().cpu().numpy()
            appearance_np = appearance.detach().cpu().numpy()
            
            if obj_id not in track_manager.tracks:
                temp_id = track_manager.init_track(box_np, appearance_np)
                track = track_manager.tracks[temp_id]
                track.track_id = obj_id
                track.kalman.track_id = obj_id
                track_manager.tracks[obj_id] = track
                del track_manager.tracks[temp_id]
            else:
                track_manager.update_track(obj_id, box_np, appearance_np)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames, {len(track_manager.tracks)} active tracks")
    
    print(f"Processed {frame_count} frames from {current_sequence}")
    print(f"Total tracks: {len(track_manager.tracks)}")
    
    track_features, track_ids = track_manager.get_track_features()
    
    print(f"Confirmed tracks: {len(track_ids)}")
    print(f"Track features shape: {track_features.shape}")
    
    if len(track_ids) > 0:
        history_lengths = [len(track_manager.tracks[tid].history) for tid in track_ids]
        print(f"Track history statistics:")
        print(f"  Mean: {np.mean(history_lengths):.1f} frames")
        print(f"  Min: {np.min(history_lengths)} frames")
        print(f"  Max: {np.max(history_lengths)} frames")
    
    return track_manager, track_features, track_ids


def visualize_track_features(track_manager, track_features, track_ids, save_path=None):
    """Visualize track feature embeddings."""
    print("\nTEST 4: Track Feature Visualization")
    
    if save_path is None:
        save_path = os.path.join(project_root, "outputs", "visualizations", "track_features.png")
    save_path = os.path.normpath(os.path.abspath(save_path))
    
    if len(track_ids) < 2:
        print("Not enough tracks for visualization")
        return
    
    features_np = track_features.detach().cpu().numpy()
    
    mean = features_np.mean(axis=0)
    centered = features_np - mean
    cov = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    features_2d = centered @ eigenvectors[:, :2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(track_ids)))
    
    for i, (track_id, color) in enumerate(zip(track_ids, colors)):
        track = track_manager.tracks[track_id]
        history_len = len(track.history)
        
        ax.scatter(features_2d[i, 0], features_2d[i, 1],
                  c=[color], s=100, alpha=0.7,
                  label=f'ID {track_id} ({history_len} frames)')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Track Feature Embeddings')
    
    if len(track_ids) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """Run all track encoder tests."""
    try:
        encoder = test_basic_track_encoder()
        test_multiple_tracks()
        track_manager, track_features, track_ids = test_real_mot17_tracks()
        
        if track_manager is not None and track_features is not None and track_ids is not None:
            output_dir = os.path.join(project_root, "outputs", "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            visualize_track_features(track_manager, track_features, track_ids)
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

