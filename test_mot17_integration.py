import sys
import os
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from model.penmot_model import PENMOT


def test_penmot_on_real_data():
    print("TEST: PENMOT on Real MOT17 Data")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    possible_paths = [
        os.path.join(project_root, "data", "MOT17"),
        os.path.join(project_root, "data", "MOT17", "MOT17"),
        os.path.join(project_root, "MOT17"),
        "./data/MOT17",
        "./data/MOT17/MOT17",
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
        print("\nPlease ensure MOT17 dataset is downloaded and structure is:")
        print("  MOT17/train/MOT17-XX-FRCNN/...")
        return False

    print(f"Found MOT17 dataset at: {MOT17_PATH}")

    print("\nLoading dataset...")
    train_loader, val_loader = create_mot17_dataloaders(
        MOT17_PATH,
        batch_size=1,
        num_workers=0
    )

    print("\nInitializing PENMOT...")
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        freeze_detection_encoder=True
    ).to(device)

    model.eval()

    print("\nProcessing first 10 frames...")
    frame_count = 0
    max_frames = 10
    current_sequence = None

    total_detections = 0
    total_tracks = 0

    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in train_loader:
            if frame_count >= max_frames:
                break

            if current_sequence is None:
                current_sequence = metadata['sequence']
            elif metadata['sequence'] != current_sequence:
                continue

            if len(boxes) == 0:
                frame_count += 1
                continue

            img = img.to(device)
            boxes = boxes.to(device)

            track_ids, features = model.track_frame(img, boxes)

            total_detections += len(boxes)
            total_tracks = len(model.track_manager.tracks)

            print(f"  Frame {frame_count + 1}: {len(boxes)} detections -> {len(track_ids)} assigned")
            print(f"    Ground truth IDs: {ids.tolist()[:5]}...")
            print(f"    Assigned IDs: {track_ids[:5]}...")
            print(f"    Active tracks: {total_tracks}")

            frame_count += 1

    print(f"\nSummary:")
    print(f"  Processed {frame_count} frames from {current_sequence}")
    print(f"  Total detections: {total_detections}")
    print(f"  Active tracks: {total_tracks}")
    print(f"  Average detections per frame: {total_detections / frame_count:.1f}")

    return True


def test_forward_backward():
    print("\n" + "=" * 60)
    print("TEST: Forward + Backward Pass")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        freeze_detection_encoder=True
    ).to(device)

    model.train()

    detection_features = torch.randn(5, 512, requires_grad=True).to(device)
    track_features = torch.randn(3, 512, requires_grad=True).to(device)

    assignment = model(detection_features, track_features)

    loss = assignment.sum()
    loss.backward()

    print(f"  Assignment shape: {assignment.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Detection features grad: {detection_features.grad is not None}")
    print(f"  Track features grad: {track_features.grad is not None}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    return True


def main():
    print("MOT17 INTEGRATION TESTS")
    print("=" * 60)

    results = {}

    try:
        results['real_data'] = test_penmot_on_real_data()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['real_data'] = False

    try:
        results['forward_backward'] = test_forward_backward()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['forward_backward'] = False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\n" + "=" * 60)
        print("READY TO TRAIN!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Monitor: tensorboard --logdir runs/penmot_training")
        print("  3. Wait for training to complete (~10 epochs)")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)