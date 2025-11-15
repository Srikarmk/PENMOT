import sys
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders


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


class SimpleIOUTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1

    def track_frame(self, boxes):
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes

        if len(self.tracks) == 0:
            track_ids = []
            for box in boxes_np:
                self.tracks[self.next_id] = box
                track_ids.append(self.next_id)
                self.next_id += 1
            return track_ids

        track_ids = [-1] * len(boxes_np)
        used_tracks = set()

        for det_idx, det_box in enumerate(boxes_np):
            best_iou = self.iou_threshold
            best_track_id = -1

            for track_id, track_box in self.tracks.items():
                if track_id in used_tracks:
                    continue

                iou = compute_iou(det_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id >= 0:
                track_ids[det_idx] = best_track_id
                used_tracks.add(best_track_id)
                self.tracks[best_track_id] = det_box

        for det_idx, det_box in enumerate(boxes_np):
            if track_ids[det_idx] == -1:
                self.tracks[self.next_id] = det_box
                track_ids[det_idx] = self.next_id
                self.next_id += 1

        return track_ids

    def reset(self):
        self.tracks = {}
        self.next_id = 1


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


def evaluate_tracker(tracker, data_loader, max_frames=200):
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

    print("Evaluating Simple IOU Tracker...")

    for img, boxes, ids, visibility, metadata in tqdm(data_loader, desc="Evaluation"):
        seq_name = metadata['sequence']

        if current_sequence is None:
            current_sequence = seq_name
        elif seq_name != current_sequence:
            if frame_count >= max_frames:
                break
            current_sequence = seq_name
            frame_count = 0
            tracker.reset()

        if frame_count >= max_frames:
            continue

        if len(boxes) == 0:
            continue

        pred_ids = tracker.track_frame(boxes)

        boxes_np = boxes.numpy()
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


def print_results(stats, metrics, tracker_name="Simple IOU Tracker"):
    print("\n" + "=" * 60)
    print(f"{tracker_name} RESULTS")
    print("=" * 60)

    print("\nOverall Metrics:")
    print(f"  MOTA: {metrics['mota'] * 100:.2f}%")
    print(f"  Recall: {metrics['recall'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  ID Switches: {metrics['id_switches']}")
    print(f"  Avg ID Switches per frame: {metrics['avg_switches_per_frame']:.2f}")

    print("\nPer-Sequence Results:")
    for seq_name, seq_stats in stats.items():
        print(f"  {seq_name}: {seq_stats['id_switches']} switches in {seq_stats['frames']} frames")


def main():
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

    print("Loading dataset...")
    _, val_loader = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)

    tracker = SimpleIOUTracker(iou_threshold=0.3)

    stats = evaluate_tracker(tracker, val_loader, max_frames=200)
    metrics = compute_metrics(stats)

    print_results(stats, metrics)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print("\nThis simple IOU tracker uses NO learning - just box overlap.")
    print("Compare PENMOT results to this baseline to see if the learned")
    print("features are actually helping or hurting performance.")


if __name__ == "__main__":
    main()