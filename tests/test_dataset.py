import sys
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.mot17_dataset import MOT17Dataset, create_mot17_dataloaders

def test_basic_loading():
    """Test basic dataset loading."""
    print("TEST 1: Basic Dataset Loading")
    
    MOT17_PATH = os.path.join(project_root, "data", "MOT17", "MOT17")
    MOT17_PATH = os.path.normpath(os.path.abspath(MOT17_PATH))
    
    train_loader, val_loader = create_mot17_dataloaders(
        MOT17_PATH, 
        batch_size=1, 
        num_workers=0
    )
    
    print(f"Train loader created: {len(train_loader)} frames")
    print(f"Val loader created: {len(val_loader)} frames")
    
    return train_loader, val_loader


def test_single_sample(train_loader):
    """Test loading a single sample."""
    print("\nTEST 2: Single Sample Loading")
    
    img, boxes, ids, visibility, metadata = next(iter(train_loader))
    
    print(f"Sample loaded from: {metadata['sequence']}")
    print(f"Frame ID: {metadata['frame_id']}")
    print(f"Image path: {metadata['img_path']}")
    print(f"Image tensor shape: {img.shape} (expected: [3, H, W])")
    
    print(f"Number of detections in this frame: {len(boxes)}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"IDs shape: {ids.shape}")
    print(f"Visibility shape: {visibility.shape}")
    
    print(f"Object IDs in frame: {ids.tolist()}")
    print(f"Visibility scores: {[f'{v:.2f}' for v in visibility.tolist()]}")
    
    print(f"First 3 bounding boxes [x1, y1, x2, y2]:")
    for i, box in enumerate(boxes[:3]):
        print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    return img, boxes, ids, visibility, metadata


def test_multiple_frames(train_loader, num_frames=5):
    """Test loading multiple frames."""
    print(f"\nTEST 3: Loading {num_frames} Frames")
    
    frame_stats = []
    
    for i, (img, boxes, ids, visibility, metadata) in enumerate(train_loader):
        if i >= num_frames:
            break
        
        frame_stats.append({
            'sequence': metadata['sequence'],
            'frame_id': metadata['frame_id'],
            'num_detections': len(boxes),
            'unique_ids': len(torch.unique(ids))
        })
    
    print(f"\n{'Sequence':<20} {'Frame':<8} {'Detections':<12} {'Unique IDs'}")
    print("-" * 60)
    for stat in frame_stats:
        print(f"{stat['sequence']:<20} {stat['frame_id']:<8} {stat['num_detections']:<12} {stat['unique_ids']}")
    
    return frame_stats


def visualize_sample(img, boxes, ids, visibility, metadata, save_path=None):
    """Visualize a sample frame with bounding boxes."""
    print("\nTEST 4: Visualization")
    
    if save_path is None:
        save_path = os.path.join(project_root, "outputs", "visualizations", "sample_frame.png")
    save_path = os.path.normpath(os.path.abspath(save_path))
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(torch.unique(ids))))
    id_to_color = {id.item(): colors[i] for i, id in enumerate(torch.unique(ids))}
    
    for box, obj_id, vis in zip(boxes, ids, visibility):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        
        color = id_to_color[obj_id.item()]
        
        rect = patches.Rectangle((x1, y1), w, h, 
                                 linewidth=2, 
                                 edgecolor=color, 
                                 facecolor='none')
        ax.add_patch(rect)
        
        label = f'ID:{obj_id.item()} ({vis:.2f})'
        ax.text(x1, y1-5, label, 
               color='white', 
               fontsize=10, 
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_title(f"{metadata['sequence']} - Frame {metadata['frame_id']}\n"
                f"Detections: {len(boxes)} | Unique IDs: {len(torch.unique(ids))}")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def test_data_consistency(train_loader, num_samples=10):
    """Test data consistency across multiple samples."""
    print("\nTEST 5: Data Consistency")
    
    all_sequences = set()
    total_detections = 0
    total_unique_ids = set()
    
    for i, (img, boxes, ids, visibility, metadata) in enumerate(train_loader):
        if i >= num_samples:
            break
        
        all_sequences.add(metadata['sequence'])
        total_detections += len(boxes)
        total_unique_ids.update(ids.tolist())
        
        assert (boxes[:, 2] > boxes[:, 0]).all(), "Invalid box: x2 <= x1"
        assert (boxes[:, 3] > boxes[:, 1]).all(), "Invalid box: y2 <= y1"
        assert (visibility >= 0).all() and (visibility <= 1).all(), "Invalid visibility"
        assert img.shape[0] == 3, "Image should have 3 channels"
    
    print(f"Tested {num_samples} samples")
    print(f"Sequences covered: {sorted(all_sequences)}")
    print(f"Total detections: {total_detections}")
    print(f"Unique object IDs seen: {len(total_unique_ids)}")
    print(f"Average detections per frame: {total_detections/num_samples:.1f}")
    print("All consistency checks passed")


def main():
    """Run all tests."""
    print("\nMOT17 DATASET TESTING")
    
    try:
        train_loader, val_loader = test_basic_loading()
        img, boxes, ids, visibility, metadata = test_single_sample(train_loader)
        frame_stats = test_multiple_frames(train_loader, num_frames=5)
        
        output_dir = os.path.join(project_root, "outputs", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        visualize_sample(img, boxes, ids, visibility, metadata)
        
        test_data_consistency(train_loader, num_samples=20)
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

