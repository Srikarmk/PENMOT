import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from PIL import Image
import cv2


def plot_embeddings(features, labels, method='pca', save_path=None, title='Feature Embeddings'):
    """
    Plot feature embeddings in 2D using PCA.
    
    Args:
        features: [N, D] feature tensor
        labels: [N] label tensor (e.g., object IDs)
        method: 'pca' or other dimensionality reduction
        save_path: Path to save figure
        title: Plot title
    """
    features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    # PCA projection
    mean = features_np.mean(axis=0, keepdims=True)
    centered = features_np - mean
    cov = (centered.T @ centered) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues.real)[::-1]
    eigenvectors = eigenvectors[:, idx]
    features_2d = (centered @ eigenvectors[:, :2]).real
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    unique_labels = np.unique(labels_np)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors[i]], label=f'ID {label}',
                  alpha=0.7, s=60, edgecolors='black', linewidths=0.5)
    
    # Explained variance
    explained_var = eigenvalues.real[:2] / eigenvalues.real.sum()
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    else:
        ax.text(1.05, 0.5, f'{len(unique_labels)} unique IDs', 
               transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_similarity_matrix(features, labels, save_path=None, title='Feature Similarity Matrix'):
    """
    Plot pairwise similarity matrix of features.
    
    Args:
        features: [N, D] feature tensor
        labels: [N] label tensor
        save_path: Path to save figure
        title: Plot title
    """
    features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    # Normalize features
    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity = features_norm @ features_norm.T
    
    # Sort by labels for better visualization
    sorted_idx = np.argsort(labels_np)
    similarity = similarity[sorted_idx][:, sorted_idx]
    sorted_labels = labels_np[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # Add grid lines between different IDs
    unique_labels, label_indices = np.unique(sorted_labels, return_index=True)
    for idx in label_indices[1:]:
        ax.axhline(y=idx-0.5, color='black', linewidth=1, alpha=0.5)
        ax.axvline(x=idx-0.5, color='black', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_intra_inter_class_distances(features, labels, save_path=None):
    """
    Plot distribution of intra-class vs inter-class distances.
    
    Args:
        features: [N, D] feature tensor
        labels: [N] label tensor
        save_path: Path to save figure
    """
    features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    # Normalize
    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise similarities
    similarity = features_norm @ features_norm.T
    
    intra_class_sims = []
    inter_class_sims = []
    
    for i in range(len(labels_np)):
        for j in range(i+1, len(labels_np)):
            sim = similarity[i, j]
            if labels_np[i] == labels_np[j]:
                intra_class_sims.append(sim)
            else:
                inter_class_sims.append(sim)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = np.linspace(-1, 1, 50)
    ax1.hist(intra_class_sims, bins=bins, alpha=0.7, label='Intra-class (same ID)', 
            color='green', edgecolor='black')
    ax1.hist(inter_class_sims, bins=bins, alpha=0.7, label='Inter-class (different ID)', 
            color='red', edgecolor='black')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Similarity Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    data = [intra_class_sims, inter_class_sims]
    bp = ax2.boxplot(data, labels=['Intra-class', 'Inter-class'], 
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Similarity Statistics', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    intra_mean = np.mean(intra_class_sims)
    inter_mean = np.mean(inter_class_sims)
    margin = intra_mean - inter_mean
    
    stats_text = f'Intra-class mean: {intra_mean:.3f}\n'
    stats_text += f'Inter-class mean: {inter_mean:.3f}\n'
    stats_text += f'Separation margin: {margin:.3f}'
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        print(f"  Intra-class mean: {intra_mean:.3f}")
        print(f"  Inter-class mean: {inter_mean:.3f}")
        print(f"  Margin: {margin:.3f}")
    
    return fig, intra_mean, inter_mean, margin


def plot_tracking_results(img, boxes, ids, predicted_boxes=None, save_path=None):
    """
    Visualize tracking results on a frame.
    
    Args:
        img: [3, H, W] image tensor or PIL Image
        boxes: [N, 4] ground truth boxes
        ids: [N] object IDs
        predicted_boxes: [N, 4] predicted boxes (optional)
        save_path: Path to save figure
    """
    # Convert image to numpy
    if torch.is_tensor(img):
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(img) / 255.0
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img_np)
    
    # Color map for IDs
    unique_ids = np.unique(ids.detach().cpu().numpy() if torch.is_tensor(ids) else ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    id_to_color = {id_val: colors[i] for i, id_val in enumerate(unique_ids)}
    
    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else boxes
    ids_np = ids.detach().cpu().numpy() if torch.is_tensor(ids) else ids
    
    for i, (box, obj_id) in enumerate(zip(boxes_np, ids_np)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        color = id_to_color[obj_id]
        
        # Draw ground truth box (solid)
        rect = patches.Rectangle((x1, y1), w, h,
                                 linewidth=2.5,
                                 edgecolor=color,
                                 facecolor='none',
                                 linestyle='-')
        ax.add_patch(rect)
        
        # Draw predicted box (dashed) if available
        if predicted_boxes is not None:
            pred_box = predicted_boxes[i]
            px1, py1, px2, py2 = pred_box
            pw, ph = px2 - px1, py2 - py1
            
            pred_rect = patches.Rectangle((px1, py1), pw, ph,
                                         linewidth=2,
                                         edgecolor=color,
                                         facecolor='none',
                                         linestyle='--',
                                         alpha=0.7)
            ax.add_patch(pred_rect)
        
        # Label
        label = f'ID {obj_id}'
        ax.text(x1, y1 - 10, label,
               color='white',
               fontsize=11,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    ax.set_title('Tracking Results' + (' (solid=GT, dashed=predicted)' if predicted_boxes is not None else ''),
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_track_history(track_manager, track_id, save_path=None):
    """
    Visualize a single track's history over time.
    
    Args:
        track_manager: TrackManager instance
        track_id: Track ID to visualize
        save_path: Path to save figure
    """
    if track_id not in track_manager.tracks:
        print(f"Track {track_id} not found")
        return None
    
    track = track_manager.tracks[track_id]
    history = track.get_history()
    
    # Extract data
    centers = []
    velocities = []
    
    for frame_data in history:
        bbox = frame_data['bbox']
        motion = frame_data['motion']
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        centers.append([cx, cy])
        
        vx, vy = motion[4], motion[5]
        velocities.append([vx, vy])
    
    centers = np.array(centers)
    velocities = np.array(velocities)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Trajectory
    ax1.plot(centers[:, 0], centers[:, 1], 'b-', linewidth=2, marker='o', markersize=5)
    ax1.scatter(centers[0, 0], centers[0, 1], c='green', s=200, marker='*', 
               label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(centers[-1, 0], centers[-1, 1], c='red', s=200, marker='*',
               label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # Add arrows for direction
    skip = max(1, len(centers) // 10)
    for i in range(0, len(centers)-1, skip):
        dx = centers[i+1, 0] - centers[i, 0]
        dy = centers[i+1, 1] - centers[i, 1]
        ax1.arrow(centers[i, 0], centers[i, 1], dx, dy,
                 head_width=10, head_length=15, fc='blue', ec='blue', alpha=0.5)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=12)
    ax1.set_ylabel('Y Position (pixels)', fontsize=12)
    ax1.set_title(f'Track {track_id} Trajectory ({len(history)} frames)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Image coordinates
    
    # Velocity over time
    frames = np.arange(len(velocities))
    ax2.plot(frames, velocities[:, 0], 'r-', label='X velocity', linewidth=2)
    ax2.plot(frames, velocities[:, 1], 'b-', label='Y velocity', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Velocity (pixels/frame)', fontsize=12)
    ax2.set_title(f'Track {track_id} Velocity Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_feature_analysis_report(detection_features, detection_ids, 
                                  track_features, track_ids,
                                  output_dir='outputs/visualizations'):
    """
    Create comprehensive feature analysis report with multiple visualizations.
    
    Args:
        detection_features: [N, D] detection feature tensor
        detection_ids: [N] detection ID tensor
        track_features: [M, D] track feature tensor
        track_ids: [M] track ID tensor
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("GENERATING FEATURE ANALYSIS REPORT")
    
    print("\n1. Detection Feature Embeddings...")
    plot_embeddings(detection_features, detection_ids,
                   save_path=f'{output_dir}/detection_embeddings.png',
                   title='Detection Feature Embeddings (Appearance)')
    
    print("\n2. Track Feature Embeddings...")
    plot_embeddings(track_features, track_ids,
                   save_path=f'{output_dir}/track_embeddings.png',
                   title='Track Feature Embeddings (Appearance + Motion + Temporal)')
    
    print("\n3. Detection Similarity Matrix...")
    plot_similarity_matrix(detection_features, detection_ids,
                          save_path=f'{output_dir}/detection_similarity.png',
                          title='Detection Feature Similarity Matrix')
    
    print("\n4. Track Similarity Matrix...")
    plot_similarity_matrix(track_features, track_ids,
                          save_path=f'{output_dir}/track_similarity.png',
                          title='Track Feature Similarity Matrix')
    
    print("\n5. Detection Intra/Inter-class Analysis...")
    _, d_intra, d_inter, d_margin = plot_intra_inter_class_distances(
        detection_features, detection_ids,
        save_path=f'{output_dir}/detection_distances.png'
    )
    
    print("\n6. Track Intra/Inter-class Analysis...")
    _, t_intra, t_inter, t_margin = plot_intra_inter_class_distances(
        track_features, track_ids,
        save_path=f'{output_dir}/track_distances.png'
    )
    
    print("\nFEATURE QUALITY SUMMARY")
    print(f"Detection Features (Appearance only):")
    print(f"  Intra-class similarity: {d_intra:.3f}")
    print(f"  Inter-class similarity: {d_inter:.3f}")
    print(f"  Separation margin: {d_margin:.3f}")
    
    print(f"Track Features (Appearance + Motion + Temporal):")
    print(f"  Intra-class similarity: {t_intra:.3f}")
    print(f"  Inter-class similarity: {t_inter:.3f}")
    print(f"  Separation margin: {t_margin:.3f}")
    
    print(f"Improvement from Temporal Encoding:")
    print(f"  Margin increase: {t_margin - d_margin:.3f}")
    
    print(f"All visualizations saved to: {output_dir}/")


def main():
    """Generate all figures for Milestone 1 report."""
    import sys
    import os
    
    # Add project root to path (from features/ directory, go up one level)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Try to import required modules
    try:
        from data.mot17_dataset import create_mot17_dataloaders
        from features.detection_encoder import DetectionEncoder
        from features.track_encoder import TrackEncoder, TrackManager
    except ImportError as e:
        print(f"Error: Required modules not found: {e}")
        print("Please ensure you're running this from the project root directory.")
        return
    
    print("MILESTONE 1 FIGURE GENERATION")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    detection_encoder = DetectionEncoder(feature_dim=512, pretrained=True).to(device)
    detection_encoder.eval()
    
    track_encoder = TrackEncoder().to(device)
    track_encoder.eval()
    
    track_manager = TrackManager(track_encoder)
    
    possible_paths = [
        "./data/MOT17/MOT17",
        "./MOT17/MOT17",
        "data/MOT17/MOT17",
    ]
    
    MOT17_PATH = None
    for path in possible_paths:
        normalized_path = os.path.normpath(os.path.abspath(path))
        if os.path.exists(normalized_path):
            train_path = os.path.join(normalized_path, 'train')
            train_path = os.path.normpath(train_path)
            if os.path.exists(train_path):
                MOT17_PATH = normalized_path
                break
    
    if MOT17_PATH is None:
        print(f"Error: MOT17 dataset not found")
        print(f"Checked paths: {possible_paths}")
        return
    
    print(f"Using MOT17 dataset at: {MOT17_PATH}")
    
    train_loader, _ = create_mot17_dataloaders(MOT17_PATH, batch_size=1, num_workers=0)
    
    print("Processing frames and extracting features...")
    
    all_detection_features = []
    all_detection_ids = []
    
    sample_img = None
    sample_boxes = None
    sample_ids = None
    
    frame_count = 0
    max_frames = 100
    current_sequence = None
    
    try:
        for img, boxes, ids, visibility, metadata in train_loader:
            # Use one sequence only
            if current_sequence is None:
                current_sequence = metadata['sequence']
            elif metadata['sequence'] != current_sequence:
                continue
            
            if frame_count >= max_frames:
                break
            
            img = img.to(device)
            boxes = boxes.to(device)
            
            # Skip if no detections
            if len(boxes) == 0:
                frame_count += 1
                continue
            
            # Extract detection features
            with torch.no_grad():
                det_features = detection_encoder(img, boxes)
            
            all_detection_features.append(det_features.cpu())
            all_detection_ids.append(ids)
            
            # Update tracks
            for i in range(len(boxes)):
                box = boxes[i]
                obj_id = ids[i].item()
                appearance = det_features[i]
                
                # Convert to numpy
                box_np = box.detach().cpu().numpy()
                appearance_np = appearance.detach().cpu().numpy()
                
                if obj_id not in track_manager.tracks:
                    # Initialize new track with the actual object ID
                    temp_id = track_manager.init_track(box_np, appearance_np)
                    track = track_manager.tracks[temp_id]
                    track.track_id = obj_id
                    track.kalman.track_id = obj_id
                    track_manager.tracks[obj_id] = track
                    del track_manager.tracks[temp_id]
                else:
                    # Update existing track
                    track_manager.update_track(obj_id, box_np, appearance_np)
            
            # Save one sample frame for visualization
            if frame_count == 30:
                sample_img = img.cpu()
                sample_boxes = boxes.cpu()
                sample_ids = ids.cpu()
            
            frame_count += 1
            
            if frame_count % 20 == 0:
                print(f"  Processed {frame_count} frames...")
        
        print(f"Processed {frame_count} frames from {current_sequence}")
        print(f"Active tracks: {len(track_manager.tracks)}")
        
        track_features, track_ids_list = track_manager.get_track_features()
        
        all_detection_features = torch.cat(all_detection_features, dim=0)
        all_detection_ids = torch.cat(all_detection_ids, dim=0)
        
        max_samples = 500
        if len(all_detection_features) > max_samples:
            indices = torch.randperm(len(all_detection_features))[:max_samples]
            all_detection_features = all_detection_features[indices]
            all_detection_ids = all_detection_ids[indices]
        
        print(f"Detection features: {all_detection_features.shape}")
        print(f"Track features: {track_features.shape}")
        
        # Use outputs/visualizations as the output directory
        output_dir = os.path.join(project_root, 'outputs', 'visualizations')
        output_dir = os.path.normpath(os.path.abspath(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGENERATING VISUALIZATIONS")
        
        create_feature_analysis_report(
            all_detection_features,
            all_detection_ids,
            track_features,
            torch.tensor(track_ids_list, dtype=torch.long),
            output_dir=output_dir
        )
        
        print("\n7. Sample Tracking Visualization...")
        if sample_img is not None:
            plot_tracking_results(
                sample_img, sample_boxes, sample_ids,
                save_path=os.path.normpath(os.path.join(output_dir, 'sample_tracking.png'))
            )
        
        print("\n8. Track History Visualizations...")
        for i, track_id in enumerate(track_ids_list[:3]):
            plot_track_history(
                track_manager, track_id,
                save_path=os.path.normpath(os.path.join(output_dir, f'track_{track_id}_history.png'))
            )
        
        print(f"Figures saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

