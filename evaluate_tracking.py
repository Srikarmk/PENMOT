import sys
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from model.penmot_model import PENMOT


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-8)


def match_detections_to_gt(pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold=0.5):
    matches = []
    matched_gt = set()

    for i, (pred_box, pred_id) in enumerate(zip(pred_boxes, pred_ids)):
        best_iou = 0
        best_gt_idx = -1

        for j, (gt_box, gt_id) in enumerate(zip(gt_boxes, gt_ids)):
            if j in matched_gt:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j

        if best_gt_idx >= 0:
            matches.append({
                'pred_id': pred_id,
                'gt_id': gt_ids[best_gt_idx],
                'iou': best_iou
            })
            matched_gt.add(best_gt_idx)

    return matches


def evaluate_tracking(model, data_loader, device, max_frames=200):
    model.eval()

    sequence_stats = defaultdict(lambda: {
        'frames': 0,
        'total_gt': 0,
        'total_pred': 0,
        'matches': 0,
        'id_switches': 0,
        'gt_to_pred_map': {}
    })

    current_sequence = None
    frame_count = 0

    print("Evaluating tracking performance...")

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in tqdm(data_loader, desc="Evaluation"):
            seq_name = metadata['sequence']

            if current_sequence is None:
                current_sequence = seq_name
            elif seq_name != current_sequence:
                if frame_count >= max_frames:
                    break
                current_sequence = seq_name
                frame_count = 0
                model.track_manager.tracks = {}

            if frame_count >= max_frames:
                continue

            if len(boxes) == 0:
                continue

            img = img.to(device)
            boxes = boxes.to(device)

            pred_ids, features = model.track_frame(img, boxes)

            if len(pred_ids) == 0:
                continue

            boxes_np = boxes.cpu().numpy()
            gt_ids_np = ids.numpy()

            matches = match_detections_to_gt(boxes_np, pred_ids, boxes_np, gt_ids_np)

            stats = sequence_stats[seq_name]
            stats['frames'] += 1
            stats['total_gt'] += len(gt_ids_np)
            stats['total_pred'] += len(pred_ids)
            stats['matches'] += len(matches)

            for match in matches:
                gt_id = match['gt_id']
                pred_id = match['pred_id']

                if gt_id in stats['gt_to_pred_map']:
                    if stats['gt_to_pred_map'][gt_id] != pred_id:
                        stats['id_switches'] += 1
                        stats['gt_to_pred_map'][gt_id] = pred_id
                else:
                    stats['gt_to_pred_map'][gt_id] = pred_id

            frame_count += 1

    return dict(sequence_stats)


def compute_metrics(stats):
    total_gt = sum(s['total_gt'] for s in stats.values())
    total_pred = sum(s['total_pred'] for s in stats.values())
    total_matches = sum(s['matches'] for s in stats.values())
    total_switches = sum(s['id_switches'] for s in stats.values())
    total_frames = sum(s['frames'] for s in stats.values())

    recall = total_matches / total_gt if total_gt > 0 else 0
    precision = total_matches / total_pred if total_pred > 0 else 0

    mota = 1 - ((total_gt - total_matches + total_switches) / total_gt) if total_gt > 0 else 0

    return {
        'recall': recall,
        'precision': precision,
        'mota': mota,
        'id_switches': total_switches,
        'total_frames': total_frames,
        'avg_switches_per_frame': total_switches / total_frames if total_frames > 0 else 0
    }


def print_results(stats, metrics):
    print("\n" + "=" * 60)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 60)

    print("\nOverall Metrics:")
    print(f"  MOTA (Multi-Object Tracking Accuracy): {metrics['mota'] * 100:.2f}%")
    print(f"  Recall: {metrics['recall'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  ID Switches: {metrics['id_switches']}")
    print(f"  Avg ID Switches per frame: {metrics['avg_switches_per_frame']:.2f}")
    print(f"  Total frames evaluated: {metrics['total_frames']}")

    print("\nPer-Sequence Results:")
    for seq_name, seq_stats in stats.items():
        seq_recall = seq_stats['matches'] / seq_stats['total_gt'] if seq_stats['total_gt'] > 0 else 0
        seq_precision = seq_stats['matches'] / seq_stats['total_pred'] if seq_stats['total_pred'] > 0 else 0

        print(f"\n  {seq_name}:")
        print(f"    Frames: {seq_stats['frames']}")
        print(f"    Ground truth objects: {seq_stats['total_gt']}")
        print(f"    Predicted tracks: {seq_stats['total_pred']}")
        print(f"    Matches: {seq_stats['matches']}")
        print(f"    Recall: {seq_recall * 100:.2f}%")
        print(f"    Precision: {seq_precision * 100:.2f}%")
        print(f"    ID switches: {seq_stats['id_switches']}")
        print(f"    Unique GT objects tracked: {len(seq_stats['gt_to_pred_map'])}")


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

    print("\nLoading trained Sinkhorn model...")
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        freeze_detection_encoder=True
    ).to(device)

    checkpoint_path = os.path.join(project_root, 'outputs', 'checkpoints', 'penmot_sinkhorn_best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    stats = evaluate_tracking(model, val_loader, device, max_frames=200)
    metrics = compute_metrics(stats)

    print_results(stats, metrics)



if __name__ == "__main__":
    main()