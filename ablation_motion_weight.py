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


def evaluate_with_motion_weight(model, data_loader, device, motion_weight, max_frames=200):
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

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in tqdm(data_loader, desc=f"Motion weight={motion_weight}",
                                                          leave=False):
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

            pred_ids, features = model.track_frame(img, boxes, motion_weight=motion_weight)

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

    total_gt = sum(s['total_gt'] for s in sequence_stats.values())
    total_matches = sum(s['matches'] for s in sequence_stats.values())
    total_switches = sum(s['id_switches'] for s in sequence_stats.values())

    mota = 1 - ((total_gt - total_matches + total_switches) / total_gt) if total_gt > 0 else 0

    return mota, total_switches


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    possible_paths = [
        os.path.join(project_root, "data", "MOT17"),
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

    checkpoint_path = os.path.join(project_root, 'outputs', 'checkpoints', 'penmot_sinkhorn_best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    print("\n" + "=" * 60)
    print("ABLATION STUDY: Motion Weight")
    print("=" * 60)
    print("\nTesting different motion weights (0.0 = pure appearance, 1.0 = pure motion)")

    motion_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = []

    for weight in motion_weights:
        model.track_manager.tracks = {}
        model.track_manager.next_id = 1

        mota, switches = evaluate_with_motion_weight(model, val_loader, device, weight, max_frames=200)
        results.append({
            'weight': weight,
            'mota': mota,
            'switches': switches
        })
        print(f"\nMotion weight: {weight:.1f}")
        print(f"  MOTA: {mota * 100:.2f}%")
        print(f"  ID switches: {switches}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n{:<15} {:<12} {:<15}".format("Motion Weight", "MOTA", "ID Switches"))
    print("-" * 45)
    for r in results:
        print("{:<15.1f} {:<12.2f}% {:<15}".format(r['weight'], r['mota'] * 100, r['switches']))

    best = max(results, key=lambda x: x['mota'])
    print("\n" + "=" * 60)
    print(f"BEST CONFIGURATION: Motion weight = {best['weight']:.1f}")
    print(f"  MOTA: {best['mota'] * 100:.2f}%")
    print(f"  ID switches: {best['switches']}")
    print("=" * 60)


if __name__ == "__main__":
    main()