import sys
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from model.penmot_model import PENMOT
from model.losses import CombinedTrackingLoss


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def create_ground_truth_assignment(current_boxes, current_ids, previous_boxes, previous_ids, device):
    if len(previous_ids) == 0 or len(current_ids) == 0:
        return torch.zeros((len(current_ids), max(1, len(previous_ids))), device=device)

    gt_assignment = torch.zeros((len(current_ids), len(previous_ids)), device=device)

    current_ids_list = current_ids.tolist() if torch.is_tensor(current_ids) else current_ids
    previous_ids_list = previous_ids.tolist() if torch.is_tensor(previous_ids) else previous_ids

    current_boxes_np = current_boxes.cpu().numpy() if torch.is_tensor(current_boxes) else current_boxes
    previous_boxes_np = previous_boxes.cpu().numpy() if torch.is_tensor(previous_boxes) else previous_boxes

    for i, curr_id in enumerate(current_ids_list):
        for j, prev_id in enumerate(previous_ids_list):
            if curr_id == prev_id:
                iou = compute_iou(current_boxes_np[i], previous_boxes_np[j])
                if iou > 0.3:
                    gt_assignment[i, j] = 1.0
                    break

    return gt_assignment


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0
    loss_components = {'assignment': 0, 'contrastive': 0, 'consistency': 0}
    num_batches = 0

    sequence_history = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for img, boxes, ids, visibility, metadata in pbar:
        if len(boxes) == 0:
            continue

        seq_name = metadata['sequence']

        if seq_name not in sequence_history:
            sequence_history[seq_name] = {
                'prev_boxes': None,
                'prev_detections': None,
                'prev_ids': None
            }

        img = img.to(device)
        boxes = boxes.to(device)

        detection_features = model.extract_detection_features(img, boxes)

        if len(detection_features) == 0:
            continue

        prev_data = sequence_history[seq_name]

        if prev_data['prev_detections'] is None:
            sequence_history[seq_name]['prev_boxes'] = boxes.detach()
            sequence_history[seq_name]['prev_detections'] = detection_features.detach()
            sequence_history[seq_name]['prev_ids'] = ids
            continue

        prev_features = prev_data['prev_detections']
        prev_ids = prev_data['prev_ids']
        prev_boxes = prev_data['prev_boxes']

        if len(prev_features) == 0:
            sequence_history[seq_name]['prev_boxes'] = boxes.detach()
            sequence_history[seq_name]['prev_detections'] = detection_features.detach()
            sequence_history[seq_name]['prev_ids'] = ids
            continue

        det_proj = model.detection_projection(detection_features)
        track_proj = model.track_projection(prev_features)

        det_out = model.detection_transformer(det_proj, track_proj)
        track_out = model.track_transformer(track_proj, det_proj)

        cost_matrix = -torch.mm(det_out, track_out.t())

        pred_assignment = model.matcher(cost_matrix)

        gt_assignment = create_ground_truth_assignment(
            boxes, ids, prev_boxes, prev_ids, device
        )

        if pred_assignment.shape != gt_assignment.shape:
            sequence_history[seq_name]['prev_boxes'] = boxes.detach()
            sequence_history[seq_name]['prev_detections'] = detection_features.detach()
            sequence_history[seq_name]['prev_ids'] = ids
            continue

        if gt_assignment.sum() == 0:
            sequence_history[seq_name]['prev_boxes'] = boxes.detach()
            sequence_history[seq_name]['prev_detections'] = detection_features.detach()
            sequence_history[seq_name]['prev_ids'] = ids
            continue

        loss, loss_dict = criterion(
            pred_assignment,
            gt_assignment,
            detection_features,
            prev_features,
            ids.to(device),
            prev_ids.to(device)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        sequence_history[seq_name]['prev_boxes'] = boxes.detach()
        sequence_history[seq_name]['prev_detections'] = detection_features.detach()
        sequence_history[seq_name]['prev_ids'] = ids

        total_loss += loss.item()
        loss_components['assignment'] += loss_dict['assignment_loss']
        loss_components['contrastive'] += loss_dict['contrastive_loss']
        loss_components['consistency'] += loss_dict['consistency_loss']
        num_batches += 1

        if num_batches % 10 == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'assign': f"{loss_dict['assignment_loss']:.4f}",
                'batches': num_batches
            })

    if num_batches == 0:
        return 0, loss_components

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    num_batches = 0

    sequence_history = {}

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in tqdm(val_loader, desc="Validation"):
            if len(boxes) == 0:
                continue

            seq_name = metadata['sequence']

            if seq_name not in sequence_history:
                sequence_history[seq_name] = {
                    'prev_boxes': None,
                    'prev_detections': None,
                    'prev_ids': None
                }

            img = img.to(device)
            boxes = boxes.to(device)

            detection_features = model.extract_detection_features(img, boxes)

            if len(detection_features) == 0:
                continue

            prev_data = sequence_history[seq_name]

            if prev_data['prev_detections'] is None:
                sequence_history[seq_name]['prev_boxes'] = boxes
                sequence_history[seq_name]['prev_detections'] = detection_features
                sequence_history[seq_name]['prev_ids'] = ids
                continue

            prev_features = prev_data['prev_detections']
            prev_ids = prev_data['prev_ids']
            prev_boxes = prev_data['prev_boxes']

            if len(prev_features) == 0:
                sequence_history[seq_name]['prev_boxes'] = boxes
                sequence_history[seq_name]['prev_detections'] = detection_features
                sequence_history[seq_name]['prev_ids'] = ids
                continue

            det_proj = model.detection_projection(detection_features)
            track_proj = model.track_projection(prev_features)

            det_out = model.detection_transformer(det_proj, track_proj)
            track_out = model.track_transformer(track_proj, det_proj)

            cost_matrix = -torch.mm(det_out, track_out.t())

            pred_assignment = model.matcher(cost_matrix)

            gt_assignment = create_ground_truth_assignment(
                boxes, ids, prev_boxes, prev_ids, device
            )

            if pred_assignment.shape != gt_assignment.shape:
                sequence_history[seq_name]['prev_boxes'] = boxes
                sequence_history[seq_name]['prev_detections'] = detection_features
                sequence_history[seq_name]['prev_ids'] = ids
                continue

            if gt_assignment.sum() == 0:
                sequence_history[seq_name]['prev_boxes'] = boxes
                sequence_history[seq_name]['prev_detections'] = detection_features
                sequence_history[seq_name]['prev_ids'] = ids
                continue

            loss, _ = criterion(
                pred_assignment,
                gt_assignment,
                detection_features,
                prev_features,
                ids.to(device),
                prev_ids.to(device)
            )

            sequence_history[seq_name]['prev_boxes'] = boxes
            sequence_history[seq_name]['prev_detections'] = detection_features
            sequence_history[seq_name]['prev_ids'] = ids

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return 0

    return total_loss / num_batches


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    possible_paths = [
        os.path.join(project_root, "data", "MOT17"),
        os.path.join(project_root, "data", "MOT17", "MOT17"),
        os.path.join(project_root, "MOT17"),
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
        print(f"Error: MOT17 dataset not found")
        return

    print("Loading dataset...")
    train_loader, val_loader = create_mot17_dataloaders(
        MOT17_PATH,
        batch_size=1,
        num_workers=0
    )

    print("\nInitializing PENMOT model...")
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        sinkhorn_iters=20,
        sinkhorn_tau=0.1,
        freeze_detection_encoder=True
    ).to(device)

    criterion = CombinedTrackingLoss(
        w_assignment=1.0,
        w_contrastive=0.5,
        w_consistency=0.3
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    writer = SummaryWriter(log_dir='runs/penmot_sinkhorn_training')

    num_epochs = 30
    best_val_loss = float('inf')

    output_dir = os.path.join(project_root, 'outputs', 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting training with Sinkhorn...")
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_assignment', train_components['assignment'], epoch)
        writer.add_scalar('Loss/train_contrastive', train_components['contrastive'], epoch)
        writer.add_scalar('Loss/train_consistency', train_components['consistency'], epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - Assignment: {train_components['assignment']:.4f}")
        print(f"    - Contrastive: {train_components['contrastive']:.4f}")
        print(f"    - Consistency: {train_components['consistency']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'penmot_sinkhorn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()