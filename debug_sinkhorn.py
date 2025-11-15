import sys
import os
import torch
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.mot17_dataset import create_mot17_dataloaders
from model.penmot_model import PENMOT


def debug_tracking():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
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
    
    _, val_loader = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)
    
    model = PENMOT(
        detection_feature_dim=512,
        track_feature_dim=512,
        transformer_dim=512,
        num_heads=8,
        num_layers=2,
        freeze_detection_encoder=True
    ).to(device)
    
    checkpoint_path = os.path.join(project_root, 'outputs', 'checkpoints', 'penmot_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("DEBUG: First 5 frames of tracking\n")
    print("="*80)
    
    frame_count = 0
    
    with torch.no_grad():
        for img, boxes, ids, visibility, metadata in val_loader:
            if frame_count >= 5:
                break
            
            if len(boxes) == 0:
                continue
            
            print(f"\nFrame {frame_count + 1}:")
            print(f"  Ground truth IDs: {ids.tolist()}")
            
            img = img.to(device)
            boxes = boxes.to(device)
            
            track_ids, features = model.track_frame(img, boxes)
            
            print(f"  Predicted IDs:     {track_ids}")
            
            matches = sum(1 for gt, pred in zip(ids.tolist(), track_ids) if gt == pred)
            print(f"  Matches: {matches}/{len(ids)}")
            
            if len(model.track_manager.tracks) > 0:
                print(f"  Active tracks: {list(model.track_manager.tracks.keys())}")
                
                if frame_count > 0:
                    detection_features = model.extract_detection_features(img, boxes)
                    
                    confirmed_tracks = {tid: t for tid, t in model.track_manager.tracks.items() 
                                       if t.is_confirmed(min_hits=1)}
                    
                    if len(confirmed_tracks) > 0:
                        active_track_ids = list(confirmed_tracks.keys())
                        track_features, feature_track_ids = model.extract_track_features(active_track_ids)
                        
                        if len(track_features) > 0:
                            assignment = model.forward(detection_features, track_features)
                            
                            print(f"\n  Assignment matrix shape: {assignment.shape}")
                            print(f"  Assignment matrix:")
                            print(f"    {assignment.cpu().numpy()}")
                            print(f"  Row sums: {assignment.sum(dim=1).cpu().numpy()}")
                            print(f"  Col sums: {assignment.sum(dim=0).cpu().numpy()}")
                            
                            num_ones = (assignment > 0.5).sum().item()
                            print(f"  Hard assignments (>0.5): {num_ones}")
            
            frame_count += 1
    
    print("\n" + "="*80)


if __name__ == "__main__":
    debug_tracking()