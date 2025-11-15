import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from model.penmot_model import PENMOT


def visualize_tracking(img, boxes, track_ids, gt_ids, frame_id, save_path):
    if torch.is_tensor(img):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(img) / 255.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    ax1.imshow(img_np)
    ax1.set_title(f'Frame {frame_id} - Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(img_np)
    ax2.set_title(f'Frame {frame_id} - PENMOT Tracking', fontsize=14, fontweight='bold')
    ax2.axis('off')

    boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
    gt_ids_np = gt_ids.cpu().numpy() if torch.is_tensor(gt_ids) else gt_ids

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for i, (box, gt_id, track_id) in enumerate(zip(boxes_np, gt_ids_np, track_ids)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        color_idx = int(gt_id) % 20
        color = colors[color_idx]

        rect_gt = patches.Rectangle((x1, y1), w, h, linewidth=2.5,
                                    edgecolor=color, facecolor='none')
        ax1.add_patch(rect_gt)
        ax1.text(x1, y1 - 10, f'ID {int(gt_id)}', color='white',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

        track_color_idx = int(track_id) % 20
        track_color = colors[track_color_idx]

        rect_pred = patches.Rectangle((x1, y1), w, h, linewidth=2.5,
                                      edgecolor=track_color, facecolor='none')
        ax2.add_patch(rect_pred)

        match = "✓" if int(gt_id) == int(track_id) else "✗"
        ax2.text(x1, y1 - 10, f'ID {int(track_id)} {match}', color='white',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=track_color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(model, data_loader, device, output_dir, max_frames=50):
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    current_sequence = None
    frame_count = 0
    sequence_results = []

    print(f"Running inference on validation set...")

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in tqdm(data_loader, desc="Inference"):
            if current_sequence is None:
                current_sequence = metadata['sequence']
                print(f"\nProcessing sequence: {current_sequence}")
            elif metadata['sequence'] != current_sequence:
                if frame_count >= max_frames:
                    break
                current_sequence = metadata['sequence']
                print(f"\nProcessing sequence: {current_sequence}")
                frame_count = 0
                model.track_manager.tracks = {}

            if frame_count >= max_frames:
                continue

            if len(boxes) == 0:
                continue

            img = img.to(device)
            boxes = boxes.to(device)

            track_ids, features = model.track_frame(img, boxes)

            gt_ids_list = ids.tolist()

            matches = sum(1 for gt, pred in zip(gt_ids_list, track_ids) if gt == pred)
            accuracy = matches / len(gt_ids_list) if len(gt_ids_list) > 0 else 0

            sequence_results.append({
                'sequence': metadata['sequence'],
                'frame': metadata['frame_id'],
                'detections': len(boxes),
                'matches': matches,
                'accuracy': accuracy
            })

            if frame_count % 10 == 0:
                save_path = os.path.join(output_dir,
                                         f"{metadata['sequence']}_frame_{metadata['frame_id']:06d}.png")
                visualize_tracking(img.cpu(), boxes.cpu(), track_ids, ids,
                                   metadata['frame_id'], save_path)

            frame_count += 1

    return sequence_results


def print_results(results):
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    total_detections = sum(r['detections'] for r in results)
    total_matches = sum(r['matches'] for r in results)
    overall_accuracy = total_matches / total_detections if total_detections > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"  Total frames: {len(results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Total matches: {total_matches}")
    print(f"  Overall ID accuracy: {overall_accuracy * 100:.2f}%")

    sequences = {}
    for r in results:
        seq = r['sequence']
        if seq not in sequences:
            sequences[seq] = {'detections': 0, 'matches': 0, 'frames': 0}
        sequences[seq]['detections'] += r['detections']
        sequences[seq]['matches'] += r['matches']
        sequences[seq]['frames'] += 1

    print(f"\nPer-Sequence Results:")
    for seq, stats in sequences.items():
        acc = stats['matches'] / stats['detections'] if stats['detections'] > 0 else 0
        print(f"  {seq}:")
        print(f"    Frames: {stats['frames']}")
        print(f"    Detections: {stats['detections']}")
        print(f"    ID accuracy: {acc * 100:.2f}%")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    possible_paths = [
        os.path.join(project_root, "data", "MOT17"),
        os.path.join(project_root, "data", "MOT17", "MOT17"),
        "./data/MOT17",
    ]

    MOT17_PATH = None
    for path in possible_paths:
        normalized_path = os.path.normpath(os.path.abspath(path))
        if os.path.exists(normalized_path):
            train_path = os.path.join(normalized_path, 'train')
            if os.path.exists(train_path):
                MOT17_PATH = normalized_path
                break

    if MOT17_PATH is None:
        print("Error: MOT17 dataset not found")
        return

    print("\nLoading dataset...")
    _, val_loader = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)

    print("\nLoading trained model...")
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        freeze_detection_encoder=True
    ).to(device)

    checkpoint_path = os.path.join(project_root, 'outputs', 'checkpoints', 'penmot_best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    output_dir = os.path.join(project_root, 'outputs', 'tracking_results')

    results = run_inference(model, val_loader, device, output_dir, max_frames=50)

    print_results(results)

    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()