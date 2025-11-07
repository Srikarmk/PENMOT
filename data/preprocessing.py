import os 
import pandas as pd
# from data.mot17_data import MOT17_PATH

# Try multiple possible paths
possible_paths = [
    "./data/MOT17/MOT17",
]

MOT17_PATH = None
for path in possible_paths:
    # Normalize path to handle mixed slashes on Windows
    normalized_path = os.path.normpath(os.path.abspath(path))
    if os.path.exists(normalized_path):
        train_path = os.path.join(normalized_path, 'train') 
        if os.path.exists(train_path):
            MOT17_PATH = normalized_path
            break

if MOT17_PATH is None:
    # Fallback to first path if none found
    MOT17_PATH = os.path.normpath(os.path.abspath(possible_paths[0]))

def parse_gt_file(gt_path):
    """
    Parse MOT17 ground truth file.
    
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    """
    df = pd.read_csv(gt_path, header=None, 
                     names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 
                            'conf', 'class', 'visibility'])
    return df

def preprocess_gt(gt_df, min_visibility=0.25, pedestrian_only=True):
    """
    Filter ground truth annotations.
    """
    # Filter for pedestrians only
    if pedestrian_only:
        gt_df = gt_df[gt_df['class'] == 1]
    
    # Filter by visibility
    gt_df = gt_df[gt_df['visibility'] >= min_visibility]
    
    # Filter by confidence (ground truth should have conf=1)
    gt_df = gt_df[gt_df['conf'] == 1]
    
    # Convert bbox to [x1, y1, x2, y2] format
    gt_df['x1'] = gt_df['bb_left']
    gt_df['y1'] = gt_df['bb_top']
    gt_df['x2'] = gt_df['bb_left'] + gt_df['bb_width']
    gt_df['y2'] = gt_df['bb_top'] + gt_df['bb_height']
    
    # Remove invalid boxes
    valid_boxes = (gt_df['bb_width'] > 0) & (gt_df['bb_height'] > 0)
    gt_df = gt_df[valid_boxes]
    
    return gt_df.reset_index(drop=True)

def get_frcnn_sequences(base_path, split='train'):
    """Get all FRCNN sequences."""
    split_path = os.path.join(base_path, split)
    # Normalize path to ensure consistent slashes
    split_path = os.path.normpath(split_path)
    all_sequences = sorted(os.listdir(split_path))
    
    # Filter only FRCNN sequences
    frcnn_sequences = [seq for seq in all_sequences if 'FRCNN' in seq]
    return frcnn_sequences

# Get all training sequences
train_sequences = get_frcnn_sequences(MOT17_PATH, 'train')

print("=" * 60)
print("MOT17-FRCNN Training Sequences")
print("=" * 60)
print(f"Total sequences: {len(train_sequences)}")
for seq in train_sequences:
    print(f"  - {seq}")

# Analyze each sequence
print("\n" + "=" * 60)
print("Sequence Statistics")
print("=" * 60)

sequence_stats = []

for seq_name in train_sequences:
    seq_path = os.path.join(MOT17_PATH, 'train', seq_name)
    seq_path = os.path.normpath(seq_path)  # Normalize to avoid mixed slashes
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    gt_path = os.path.normpath(gt_path)  # Normalize to avoid mixed slashes
    
    # Parse and preprocess
    gt_df = pd.read_csv(gt_path, header=None, 
                        names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 
                               'conf', 'class', 'visibility'])
    
    # Apply same preprocessing
    gt_clean = gt_df[(gt_df['class'] == 1) & 
                     (gt_df['conf'] == 1) & 
                     (gt_df['visibility'] >= 0.25)]
    
    num_frames = gt_clean['frame'].max()
    num_ids = gt_clean['id'].nunique()
    num_annotations = len(gt_clean)
    
    sequence_stats.append({
        'sequence': seq_name,
        'frames': num_frames,
        'ids': num_ids,
        'annotations': num_annotations
    })
    
    print(f"{seq_name:20s} | Frames: {num_frames:4d} | IDs: {num_ids:3d} | Boxes: {num_annotations:5d}")



# Test on one sequence
seq_path = os.path.join(MOT17_PATH, 'train', 'MOT17-02-FRCNN')
seq_path = os.path.normpath(seq_path)  # Normalize to avoid mixed slashes
gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
gt_path = os.path.normpath(gt_path)  # Normalize to avoid mixed slashes

gt_df = parse_gt_file(gt_path)

print("\nGround Truth Statistics:")
print(f"Total annotations: {len(gt_df)}")
print(f"Number of frames: {gt_df['frame'].max()}")
print(f"Number of unique IDs: {gt_df['id'].nunique()}")
print(f"Classes present: {gt_df['class'].unique()}")
print(f"\nFirst few rows:")
print(gt_df.head(10))

# Visualize detections per frame
import matplotlib.pyplot as plt
detections_per_frame = gt_df.groupby('frame').size()
plt.figure(figsize=(10, 4))
plt.plot(detections_per_frame.values)
plt.xlabel('Frame')
plt.ylabel('Number of Objects')
plt.title('Objects per Frame')

# Save to outputs/visualizations
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'visualizations')
output_dir = os.path.normpath(output_dir)
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'objects_per_frame.png')
save_path = os.path.normpath(save_path)
plt.savefig(save_path)
print(f"\nSaved: {save_path}")

# Apply preprocessing
gt_clean = preprocess_gt(gt_df)

print(f"\nAfter preprocessing:")
print(f"Remaining annotations: {len(gt_clean)}")
print(f"Unique IDs: {gt_clean['id'].nunique()}")



stats_df = pd.DataFrame(sequence_stats)
print("\n" + "=" * 60)
print(f"TOTAL: {stats_df['annotations'].sum()} annotations across {len(train_sequences)} sequences")
print("=" * 60)



# Common splits for MOT17
TRAIN_SEQUENCES = [
    'MOT17-02-FRCNN',
    'MOT17-04-FRCNN', 
    'MOT17-05-FRCNN',
    'MOT17-09-FRCNN',
    'MOT17-10-FRCNN'
]

VAL_SEQUENCES = [
    'MOT17-11-FRCNN',
    'MOT17-13-FRCNN'
]

print("Train/Val Split:")
print(f"\nTrain sequences ({len(TRAIN_SEQUENCES)}):")
for seq in TRAIN_SEQUENCES:
    stats = stats_df[stats_df['sequence'] == seq].iloc[0]
    print(f"  {seq:20s} | {stats['frames']} frames | {stats['ids']} IDs")

print(f"\nVal sequences ({len(VAL_SEQUENCES)}):")
for seq in VAL_SEQUENCES:
    stats = stats_df[stats_df['sequence'] == seq].iloc[0]
    print(f"  {seq:20s} | {stats['frames']} frames | {stats['ids']} IDs")

train_total = stats_df[stats_df['sequence'].isin(TRAIN_SEQUENCES)]['annotations'].sum()
val_total = stats_df[stats_df['sequence'].isin(VAL_SEQUENCES)]['annotations'].sum()
print(f"\nTrain annotations: {train_total}")
print(f"Val annotations: {val_total}")
print(f"Split ratio: {train_total/(train_total+val_total)*100:.1f}% / {val_total/(train_total+val_total)*100:.1f}%")