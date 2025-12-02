import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class MOT17Dataset(Dataset):
    def __init__(self, mot17_root, sequences, min_visibility=0.25, transform=None):
        self.mot17_root = os.path.normpath(os.path.abspath(mot17_root))
        self.sequences = sequences
        self.min_visibility = min_visibility
        self.transform = transform

        self.samples = self._load_annotations()

        print(f"Loaded {len(self.samples)} frames from {len(sequences)} sequences")

    def _load_annotations(self):
        samples = []

        for seq_name in self.sequences:
            seq_path = os.path.join(self.mot17_root, 'train', seq_name)
            seq_path = os.path.normpath(seq_path)
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            gt_path = os.path.normpath(gt_path)
            img_dir = os.path.join(seq_path, 'img1')
            img_dir = os.path.normpath(img_dir)

            gt_df = pd.read_csv(gt_path, header=None,
                                names=['frame', 'id', 'bb_left', 'bb_top',
                                       'bb_width', 'bb_height', 'conf', 'class', 'visibility'])

            gt_df = gt_df[(gt_df['class'] == 1) &
                          (gt_df['conf'] == 1) &
                          (gt_df['visibility'] >= self.min_visibility)]

            gt_df['x1'] = gt_df['bb_left']
            gt_df['y1'] = gt_df['bb_top']
            gt_df['x2'] = gt_df['bb_left'] + gt_df['bb_width']
            gt_df['y2'] = gt_df['bb_top'] + gt_df['bb_height']

            for frame_id, frame_data in gt_df.groupby('frame'):
                img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
                img_path = os.path.normpath(img_path)

                if not os.path.exists(img_path):
                    continue

                samples.append({
                    'sequence': seq_name,
                    'frame_id': int(frame_id),
                    'img_path': img_path,
                    'boxes': frame_data[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32),
                    'ids': frame_data['id'].values.astype(np.int64),
                    'visibility': frame_data['visibility'].values.astype(np.float32)
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample['img_path']).convert('RGB')

        boxes = torch.from_numpy(sample['boxes'])
        ids = torch.from_numpy(sample['ids'])
        visibility = torch.from_numpy(sample['visibility'])

        if self.transform:
            img = self.transform(img)

        metadata = {
            'sequence': sample['sequence'],
            'frame_id': sample['frame_id'],
            'img_path': sample['img_path']
        }

        return img, boxes, ids, visibility, metadata


def create_mot17_dataloaders(mot17_root, batch_size=1, num_workers=4):
    train_sequences = [
        'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
        'MOT17-09-FRCNN', 'MOT17-10-FRCNN'
    ]

    val_sequences = [
        'MOT17-11-FRCNN', 'MOT17-13-FRCNN'
    ]

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MOT17Dataset(mot17_root, train_sequences, transform=transform)
    val_dataset = MOT17Dataset(mot17_root, val_sequences, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=lambda x: x[0]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=lambda x: x[0]
    )

    return train_loader, val_loader