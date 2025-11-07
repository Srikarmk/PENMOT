import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from features.kalman_filter import KalmanFilter, KalmanTracker

# Try to import MOT17 dataset (optional - only needed for test 2)
try:
    from data.mot17_dataset import create_mot17_dataloaders
    MOT17_AVAILABLE = True
except (ImportError, OSError) as e:
    MOT17_AVAILABLE = False
    print(f"Warning: MOT17 dataset not available ({e}). Skipping test 2.")


def test_single_track():
    """Test Kalman Filter on a single track."""
    print("TEST 1: Single Track Prediction")
    
    true_positions = []
    measurements = []
    
    for t in range(20):
        true_x = 100 + t * 5
        true_y = 200 + t * 3
        true_w = 50
        true_h = 100
        
        noise = np.random.randn(4) * 2
        measured = np.array([
            true_x - true_w/2 + noise[0],
            true_y - true_h/2 + noise[1],
            true_x + true_w/2 + noise[2],
            true_y + true_h/2 + noise[3]
        ])
        
        true_positions.append([true_x - true_w/2, true_y - true_h/2, 
                              true_x + true_w/2, true_y + true_h/2])
        measurements.append(measured)
    
    kf = KalmanFilter(measurements[0], track_id=1)
    predictions = []
    
    for i, meas in enumerate(measurements[1:], 1):
        pred = kf.predict()
        predictions.append(pred)
        kf.update(meas)
        
        if i % 5 == 0:
            vel = kf.get_velocity()
            print(f"Frame {i}: Velocity = [{vel[0]:.2f}, {vel[1]:.2f}] pixels/frame")
    
    vel = kf.get_velocity()
    print(f"Tracked {len(measurements)} frames")
    print(f"Final velocity: [{vel[0]:.2f}, {vel[1]:.2f}] pixels/frame (expected: [5.0, 3.0])")
    
    return true_positions, measurements, predictions


def test_real_mot17_track(max_frames=50):
    """
    Test Kalman Filter on real MOT17 data.
    
    Args:
        max_frames: Maximum number of frames to collect for testing
    """
    print("\nTEST 2: Real MOT17 Track")
    
    if not MOT17_AVAILABLE:
        print("Skipping MOT17 test - dataset module not available")
        return None, None, None, None, None
    
    MOT17_PATH = os.path.join(project_root, "data", "MOT17", "MOT17")
    MOT17_PATH = os.path.normpath(os.path.abspath(MOT17_PATH))
    train_loader, _ = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)
    
    sequence_frames = []
    target_sequence = None
    frame_count = 0
    
    for img, boxes, ids, visibility, metadata in train_loader:
        if target_sequence is None:
            target_sequence = metadata['sequence']
        
        if metadata['sequence'] != target_sequence:
            if len(sequence_frames) >= max_frames:
                break
            continue
        
        if len(boxes) == 0:
            continue
        
        sequence_frames.append({
            'frame_id': metadata['frame_id'],
            'boxes': boxes.numpy(),
            'ids': ids.numpy()
        })
        
        frame_count += 1
        if len(sequence_frames) >= max_frames:
            break
    
    if len(sequence_frames) == 0:
        print(f"No frames collected from {target_sequence}")
        return None, None, None, None, None
    
    print(f"Loaded {len(sequence_frames)} frames from {target_sequence}")
    
    all_ids = []
    for frame in sequence_frames:
        all_ids.extend(frame['ids'].tolist())
    
    from collections import Counter
    id_counts = Counter(all_ids)
    target_id = id_counts.most_common(1)[0][0]
    
    print(f"Tracking object ID {target_id} (appears in {id_counts[target_id]} frames)")
    
    kf = None
    gt_boxes = []
    pred_boxes = []
    frame_numbers = []
    
    for i, frame in enumerate(sequence_frames):
        mask = frame['ids'] == target_id
        
        if not mask.any():
            if kf is not None:
                pred = kf.predict()
            continue
        
        gt_box = frame['boxes'][mask][0]
        frame_numbers.append(frame['frame_id'])
        
        if kf is None:
            kf = KalmanFilter(gt_box, track_id=target_id)
            gt_boxes.append(gt_box)
            pred_boxes.append(gt_box)
        else:
            pred = kf.predict()
            pred_boxes.append(pred)
            gt_boxes.append(gt_box)
            kf.update(gt_box)
    
    if len(gt_boxes) < 2:
        print(f"Need at least 2 frames with object {target_id} to compute prediction error")
        print(f"Found {len(gt_boxes)} frames with object {target_id}")
        return None, None, None, None, None
    
    gt_boxes = np.array(gt_boxes)
    pred_boxes = np.array(pred_boxes)
    
    gt_centers = (gt_boxes[1:, :2] + gt_boxes[1:, 2:]) / 2
    pred_centers = (pred_boxes[1:, :2] + pred_boxes[1:, 2:]) / 2
    errors = np.linalg.norm(gt_centers - pred_centers, axis=1)
    
    def bbox_iou(box1, box2):
        """Compute IoU between two boxes."""
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    ious = [bbox_iou(gt_boxes[i+1], pred_boxes[i+1]) for i in range(len(errors))]
    ious = np.array(ious)
    
    print(f"\nPrediction Error Statistics (based on {len(errors)} predictions):")
    print(f"  Mean center error: {errors.mean():.2f} pixels")
    print(f"  Median center error: {np.median(errors):.2f} pixels")
    print(f"  Max center error: {errors.max():.2f} pixels")
    print(f"  Mean IoU: {ious.mean():.3f}")
    print(f"  Min IoU: {ious.min():.3f}")
    
    return gt_boxes, pred_boxes, frame_numbers, target_id, target_sequence


def visualize_predictions(true_pos, measurements, predictions, save_path=None):
    """Visualize Kalman Filter predictions."""
    print("\nTEST 3: Visualization")
    
    if save_path is None:
        save_path = os.path.join(project_root, "outputs", "visualizations", "kalman_simulation.png")
    save_path = os.path.normpath(os.path.abspath(save_path))
    
    true_pos = np.array(true_pos)
    measurements = np.array(measurements)
    predictions = np.array(predictions)
    
    true_centers = (true_pos[:, :2] + true_pos[:, 2:]) / 2
    meas_centers = (measurements[:, :2] + measurements[:, 2:]) / 2
    pred_centers = (predictions[:, :2] + predictions[:, 2:]) / 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(true_centers[:, 0], true_centers[:, 1], 'g-', linewidth=2, label='Ground Truth')
    ax1.scatter(meas_centers[:, 0], meas_centers[:, 1], c='b', s=30, alpha=0.5, label='Noisy Measurements')
    ax1.plot(pred_centers[:, 0], pred_centers[:, 1], 'r--', linewidth=2, label='Kalman Predictions')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Object Trajectory - Kalman Filter Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    errors = np.linalg.norm(true_centers[1:] - pred_centers, axis=1)
    ax2.plot(errors, 'r-', linewidth=2)
    ax2.axhline(y=errors.mean(), color='b', linestyle='--', label=f'Mean: {errors.mean():.2f}px')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Prediction Error (pixels)')
    ax2.set_title('Kalman Filter Prediction Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main():
    """Run all Kalman Filter tests."""
    try:
        true_pos, measurements, predictions = test_single_track()
        gt_boxes, pred_boxes, frame_nums, target_id, sequence = test_real_mot17_track()
        
        output_dir = os.path.join(project_root, "outputs", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        visualize_predictions(true_pos, measurements, predictions)
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

