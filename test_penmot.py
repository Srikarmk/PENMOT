import sys
import os
import torch
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from model.set_transformer import SetTransformer
from model.sinkhorn import SinkhornMatcher, hungarian_algorithm
from model.penmot_model import PENMOT
from model.losses import CombinedTrackingLoss


def test_set_transformer():
    print("TEST 1: Set Transformer Permutation Equivariance")

    transformer = SetTransformer(d_model=512, num_heads=8, num_layers=2)
    transformer.eval()

    query = torch.randn(5, 512)
    key = torch.randn(3, 512)

    out1 = transformer(query, key)

    perm_indices = torch.randperm(5)
    query_perm = query[perm_indices]

    out2 = transformer(query_perm, key)
    out2_unperm = out2[torch.argsort(perm_indices)]

    diff = (out1 - out2_unperm).abs().max().item()

    print(f"  Input: {query.shape}, Output: {out1.shape}")
    print(f"  Max difference after permutation: {diff:.6f}")
    print(f"  Equivariant: {'YES' if diff < 1e-5 else 'NO'}")

    return diff < 1e-5


def test_sinkhorn():
    print("\nTEST 2: Sinkhorn vs Hungarian")

    cost_matrix = torch.randn(4, 6)

    matcher = SinkhornMatcher(n_iters=10, tau=0.1)
    sinkhorn_assignment = matcher(cost_matrix)

    print(f"  Cost matrix: {cost_matrix.shape}")
    print(f"  Sinkhorn assignment: {sinkhorn_assignment.shape}")
    print(f"  Row sums: {sinkhorn_assignment.sum(dim=1).tolist()}")
    print(f"  Col sums: {sinkhorn_assignment.sum(dim=0).tolist()}")

    hungarian_assignment = hungarian_algorithm(cost_matrix)
    print(f"  Hungarian assignment: {hungarian_assignment.shape}")
    print(f"  Num matches (Sinkhorn): {(sinkhorn_assignment > 0.5).sum().item()}")
    print(f"  Num matches (Hungarian): {hungarian_assignment.sum().item()}")

    return True


def test_losses():
    print("\nTEST 3: Loss Functions")

    criterion = CombinedTrackingLoss()

    pred_assignment = torch.softmax(torch.randn(5, 3), dim=1)
    gt_assignment = torch.zeros(5, 3)
    gt_assignment[0, 0] = 1.0
    gt_assignment[1, 1] = 1.0
    gt_assignment[2, 2] = 1.0

    detection_features = torch.randn(5, 512)
    track_features = torch.randn(3, 512)
    current_ids = torch.tensor([1, 2, 3, 4, 5])
    previous_ids = torch.tensor([1, 2, 3])

    loss, loss_dict = criterion(
        pred_assignment, gt_assignment,
        detection_features, track_features,
        current_ids, previous_ids
    )

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Assignment loss: {loss_dict['assignment_loss']:.4f}")
    print(f"  Contrastive loss: {loss_dict['contrastive_loss']:.4f}")
    print(f"  Consistency loss: {loss_dict['consistency_loss']:.4f}")

    return loss.item() > 0


def test_penmot_forward():
    print("\nTEST 4: PENMOT Forward Pass")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2
    ).to(device)

    model.eval()

    detection_features = torch.randn(5, 512).to(device)
    track_features = torch.randn(3, 512).to(device)

    with torch.no_grad():
        assignment = model(detection_features, track_features)

    print(f"  Detection features: {detection_features.shape}")
    print(f"  Track features: {track_features.shape}")
    print(f"  Assignment matrix: {assignment.shape}")
    print(f"  Assignment values range: [{assignment.min():.3f}, {assignment.max():.3f}]")

    matched = (assignment > 0.5).sum().item()
    print(f"  Matched pairs (>0.5): {matched}")

    return assignment.shape == (5, 3)


def test_penmot_tracking():
    print("\nTEST 5: PENMOT Tracking on Dummy Data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PENMOT().to(device)

    img = torch.randn(3, 640, 480).to(device)
    boxes1 = torch.tensor([[100, 100, 200, 300], [300, 200, 400, 400]], dtype=torch.float32).to(device)

    track_ids1, features1 = model.track_frame(img, boxes1)
    print(f"  Frame 1: {len(boxes1)} detections -> {len(track_ids1)} tracks")
    print(f"  Track IDs: {track_ids1}")

    boxes2 = torch.tensor([[105, 105, 205, 305], [305, 205, 405, 405], [500, 100, 600, 200]], dtype=torch.float32).to(
        device)

    track_ids2, features2 = model.track_frame(img, boxes2)
    print(f"  Frame 2: {len(boxes2)} detections -> {len(track_ids2)} tracks")
    print(f"  Track IDs: {track_ids2}")

    print(f"  Active tracks: {len(model.track_manager.tracks)}")

    return len(track_ids1) == 2 and len(track_ids2) == 3


def main():
    print("PENMOT COMPONENT TESTS")
    print("=" * 60)

    results = {}

    try:
        results['set_transformer'] = test_set_transformer()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['set_transformer'] = False

    try:
        results['sinkhorn'] = test_sinkhorn()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['sinkhorn'] = False

    try:
        results['losses'] = test_losses()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['losses'] = False

    try:
        results['penmot_forward'] = test_penmot_forward()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['penmot_forward'] = False

    try:
        results['penmot_tracking'] = test_penmot_tracking()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['penmot_tracking'] = False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)