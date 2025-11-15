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


def create_ground_truth_assignment(current_ids, previous_ids, device):
    if len(previous_ids) == 0 or len(current_ids) == 0:
        return torch.zeros((len(current_ids), max(1, len(previous_ids))), device=device)

    gt_assignment = torch.zeros((len(current_ids), len(previous_ids)), device=device)

    for i, curr_id in enumerate(current_ids):
        for j, prev_id in enumerate(previous_ids):
            if curr_id == prev_id:
                gt_assignment[i, j] = 1.0
                break

    return gt_assignment


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0
    loss_components = {'assignment': 0, 'contrastive': 0, 'consistency': 0}
    num_batches = 0

    sequence_tracks = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for img, boxes, ids, visibility, metadata in pbar:
        if len(boxes) == 0:
            continue

        seq_name = metadata['sequence']
        frame_id = metadata['frame_id']

        if seq_name not in sequence_tracks:
            sequence_tracks[seq_name] = {
                'track_manager': model.track_manager,
                'previous_ids': []
            }

        img = img.to(device)
        boxes = boxes.to(device)

        detection_features = model.extract_detection_features(img, boxes)

        previous_ids = sequence_tracks[seq_name]['previous_ids']

        if len(previous_ids) == 0:
            for i in range(len(boxes)):
                box = boxes[i].detach().cpu().numpy()
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = model.track_manager.init_track(box, appearance)

            sequence_tracks[seq_name]['previous_ids'] = ids.tolist()
            continue

        track_features, active_track_ids = model.extract_track_features()

        if len(track_features) == 0:
            sequence_tracks[seq_name]['previous_ids'] = ids.tolist()
            continue

        pred_assignment = model(detection_features, track_features)

        gt_assignment = create_ground_truth_assignment(
            ids.tolist(),
            active_track_ids,
            device
        )

        if pred_assignment.shape != gt_assignment.shape:
            sequence_tracks[seq_name]['previous_ids'] = ids.tolist()
            continue

        loss, loss_dict = criterion(
            pred_assignment,
            gt_assignment,
            detection_features,
            track_features,
            torch.tensor(ids.tolist(), device=device),
            torch.tensor(active_track_ids, device=device)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        matched_indices = (pred_assignment > 0.5).nonzero(as_tuple=False)
        for det_idx, track_idx in matched_indices:
            if track_idx < len(active_track_ids):
                track_id = active_track_ids[track_idx.item()]
                box = boxes[det_idx].detach().cpu().numpy()
                appearance = detection_features[det_idx].detach().cpu().numpy()
                model.track_manager.update_track(track_id, box, appearance)

        sequence_tracks[seq_name]['previous_ids'] = ids.tolist()

        total_loss += loss.item()
        loss_components['assignment'] += loss_dict['assignment_loss']
        loss_components['contrastive'] += loss_dict['contrastive_loss']
        loss_components['consistency'] += loss_dict['consistency_loss']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'assign': f"{loss_dict['assignment_loss']:.4f}",
            'contrast': f"{loss_dict['contrastive_loss']:.4f}"
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

    sequence_tracks = {}

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in tqdm(val_loader, desc="Validation"):
            if len(boxes) == 0:
                continue

            seq_name = metadata['sequence']

            if seq_name not in sequence_tracks:
                sequence_tracks[seq_name] = {
                    'previous_ids': []
                }

            img = img.to(device)
            boxes = boxes.to(device)

            detection_features = model.extract_detection_features(img, boxes)

            previous_ids = sequence_tracks[seq_name]['previous_ids']

            if len(previous_ids) == 0:
                sequence_tracks[seq_name]['previous_ids'] = ids.tolist()
                continue

            track_features, active_track_ids = model.extract_track_features()

            if len(track_features) == 0:
                continue

            pred_assignment = model(detection_features, track_features)

            gt_assignment = create_ground_truth_assignment(
                ids.tolist(),
                active_track_ids,
                device
            )

            if pred_assignment.shape != gt_assignment.shape:
                continue

            loss, _ = criterion(
                pred_assignment,
                gt_assignment,
                detection_features,
                track_features,
                torch.tensor(ids.tolist(), device=device),
                torch.tensor(active_track_ids, device=device)
            )

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
        print(f"Checked paths:")
        for path in possible_paths:
            print(f"  - {os.path.normpath(os.path.abspath(path))}")
        return

    print("Loading dataset...")
    train_loader, val_loader = create_mot17_dataloaders(
        MOT17_PATH,
        batch_size=32,
        num_workers=0
    )

    print("\nInitializing PENMOT model...")
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        sinkhorn_iters=5,
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

    writer = SummaryWriter(log_dir='runs/penmot_training')

    num_epochs = 10
    best_val_loss = float('inf')

    output_dir = os.path.join(project_root, 'outputs', 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting training...")
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'penmot_best.pth')
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