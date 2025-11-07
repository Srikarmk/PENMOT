import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from features.detection_encoder import DetectionEncoder


def test_feature_extractor():
    """Test the detection feature extractor."""
    print("TEST: Detection Feature Extraction")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder = DetectionEncoder(feature_dim=512, pretrained=True, freeze_backbone=True)
    encoder = encoder.to(device)
    encoder.eval()
    
    MOT17_PATH = os.path.join(project_root, "data", "MOT17", "MOT17")
    MOT17_PATH = os.path.normpath(os.path.abspath(MOT17_PATH))
    train_loader, _ = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)
    
    img, boxes, ids, visibility, metadata = next(iter(train_loader))
    img = img.to(device)
    boxes = boxes.to(device)
    
    print(f"Loaded frame from: {metadata['sequence']}, Frame {metadata['frame_id']}")
    print(f"Image shape: {img.shape}")
    print(f"Number of detections: {len(boxes)}")
    
    with torch.no_grad():
        features = encoder(img, boxes)
    
    print(f"Feature shape: {features.shape} (expected: [{len(boxes)}, 512])")
    print(f"Feature norm: {features.norm(dim=1).mean().item():.3f}")
    
    print(f"Feature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")
    
    return encoder, train_loader


def test_multiple_frames(encoder, train_loader, num_frames=10):
    """Extract features from multiple frames."""
    print(f"\nTEST: Extract Features from {num_frames} Frames")
    
    device = next(encoder.parameters()).device
    all_features = []
    all_ids = []
    
    for i, (img, boxes, ids, visibility, metadata) in enumerate(train_loader):
        if i >= num_frames:
            break
        
        img = img.to(device)
        boxes = boxes.to(device)
        
        with torch.no_grad():
            features = encoder(img, boxes)
        
        all_features.append(features.cpu())
        all_ids.append(ids)
        
        print(f"Frame {i+1}: {len(boxes)} detections → features {features.shape}")
    
    all_features = torch.cat(all_features, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    print(f"Total features extracted: {all_features.shape}")
    print(f"Unique object IDs: {len(torch.unique(all_ids))}")
    
    return all_features, all_ids


def compute_feature_similarity(encoder, train_loader):
    """Compute intra-class vs inter-class feature similarity."""
    print("\nTEST: Feature Similarity Analysis")
    
    device = next(encoder.parameters()).device
    
    features_dict = {}
    
    for i, (img, boxes, ids, visibility, metadata) in enumerate(train_loader):
        if i >= 5:
            break
        
        img = img.to(device)
        boxes = boxes.to(device)
        
        with torch.no_grad():
            features = encoder(img, boxes)
        
        for feat, obj_id in zip(features, ids):
            obj_id = obj_id.item()
            if obj_id not in features_dict:
                features_dict[obj_id] = []
            features_dict[obj_id].append(feat.cpu())
    
    multi_appearance_ids = {k: v for k, v in features_dict.items() if len(v) >= 2}
    
    print(f"IDs appearing multiple times: {len(multi_appearance_ids)}")
    
    if len(multi_appearance_ids) == 0:
        print("Not enough repeated IDs in these frames. Try more frames.")
        return
    
    intra_similarities = []
    for obj_id, feats in multi_appearance_ids.items():
        feats = torch.stack(feats)
        sim_matrix = torch.mm(feats, feats.t())
        mask = ~torch.eye(len(feats), dtype=torch.bool)
        intra_sim = sim_matrix[mask].mean()
        intra_similarities.append(intra_sim.item())
    
    all_ids = list(multi_appearance_ids.keys())
    inter_similarities = []
    for i in range(len(all_ids)):
        for j in range(i+1, len(all_ids)):
            feats_i = torch.stack(multi_appearance_ids[all_ids[i]])
            feats_j = torch.stack(multi_appearance_ids[all_ids[j]])
            feat_i = feats_i.mean(dim=0, keepdim=True)
            feat_j = feats_j.mean(dim=0, keepdim=True)
            sim = torch.mm(feat_i, feat_j.t()).item()
            inter_similarities.append(sim)
    
    print(f"Intra-class similarity (same ID): {np.mean(intra_similarities):.3f} ± {np.std(intra_similarities):.3f}")
    print(f"Inter-class similarity (diff ID): {np.mean(inter_similarities):.3f} ± {np.std(inter_similarities):.3f}")
    print(f"Separation margin: {np.mean(intra_similarities) - np.mean(inter_similarities):.3f}")


def visualize_embeddings_pytorch(all_features, all_ids, save_path=None):
    """Visualize feature embeddings with PCA (pure PyTorch, no sklearn)."""
    print("\nTEST: PCA Visualization (PyTorch)")
    
    if save_path is None:
        save_path = os.path.join(project_root, "outputs", "visualizations", "feature_embeddings.png")
    save_path = os.path.normpath(os.path.abspath(save_path))
    
    mean = all_features.mean(dim=0, keepdim=True)
    centered = all_features - mean
    
    cov = (centered.T @ centered) / (len(centered) - 1)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]
    
    features_2d = centered @ eigenvectors[:, :2]
    features_2d = features_2d.numpy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_ids = torch.unique(all_ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
    
    for i, obj_id in enumerate(unique_ids):
        mask = all_ids == obj_id
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=f'ID {obj_id.item()}', 
                  alpha=0.6, s=50)
    
    ax.set_title('PCA Visualization of Detection Features')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    if len(unique_ids) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

def main():
    """Run all feature extraction tests."""
    try:
        encoder, train_loader = test_feature_extractor()
        all_features, all_ids = test_multiple_frames(encoder, train_loader, num_frames=10)
        compute_feature_similarity(encoder, train_loader)
        visualize_embeddings_pytorch(all_features, all_ids)
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

