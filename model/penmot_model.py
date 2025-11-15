import torch
import torch.nn as nn
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from features.detection_encoder import DetectionEncoder
from features.track_encoder import TrackEncoder, TrackManager
from model.set_transformer import SetTransformer
from model.sinkhorn import SinkhornMatcher


class PENMOT(nn.Module):
    def __init__(self,
                 detection_feature_dim=512,
                 track_feature_dim=512,
                 transformer_dim=512,
                 num_heads=8,
                 num_layers=3,
                 sinkhorn_iters=5,
                 sinkhorn_tau=0.1,
                 freeze_detection_encoder=False):
        super().__init__()

        self.detection_encoder = DetectionEncoder(
            feature_dim=detection_feature_dim,
            pretrained=True,
            freeze_backbone=freeze_detection_encoder
        )

        self.track_encoder = TrackEncoder(
            appearance_dim=detection_feature_dim,
            motion_dim=8,
            hidden_dim=256,
            output_dim=track_feature_dim
        )

        self.detection_projection = nn.Linear(detection_feature_dim, transformer_dim)
        self.track_projection = nn.Linear(track_feature_dim, transformer_dim)

        self.detection_transformer = SetTransformer(
            d_model=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.track_transformer = SetTransformer(
            d_model=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.matcher = SinkhornMatcher(
            n_iters=sinkhorn_iters,
            tau=sinkhorn_tau
        )

        self.track_manager = TrackManager(self.track_encoder)

        print(f"[OK] PENMOT initialized")

    def extract_detection_features(self, img, boxes):
        if len(boxes) == 0:
            return torch.zeros((0, 512), device=img.device)

        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 > x1 + 10 and y2 > y1 + 10:
                valid_boxes.append(box)

        if len(valid_boxes) == 0:
            return torch.zeros((0, 512), device=img.device)

        valid_boxes = torch.stack(valid_boxes)
        return self.detection_encoder(img, valid_boxes)

    def extract_track_features(self, track_ids=None):
        return self.track_manager.get_track_features(track_ids)

    def forward(self, detection_features, track_features):
        if len(detection_features) == 0 or len(track_features) == 0:
            return torch.zeros((len(detection_features), len(track_features)))

        det_proj = self.detection_projection(detection_features)
        track_proj = self.track_projection(track_features)

        det_out = self.detection_transformer(det_proj, track_proj)
        track_out = self.track_transformer(track_proj, det_proj)

        cost_matrix = -torch.mm(det_out, track_out.t())

        assignment = self.matcher(cost_matrix)

        return assignment

    def update_tracks(self, boxes, detection_features, assignment, threshold=0.5):
        matched_tracks = []

        for det_idx in range(len(boxes)):
            if len(assignment.shape) == 2:
                track_idx = torch.argmax(assignment[det_idx])
                confidence = assignment[det_idx, track_idx]
            else:
                track_idx = torch.argmax(assignment)
                confidence = assignment[track_idx]

            if confidence > threshold:
                matched_tracks.append(track_idx.item())
            else:
                matched_tracks.append(-1)

        return matched_tracks

    def init_new_tracks(self, boxes, detection_features, unmatched_indices):
        new_track_ids = []
        for idx in unmatched_indices:
            box = boxes[idx].detach().cpu().numpy()
            appearance = detection_features[idx].detach().cpu().numpy()
            track_id = self.track_manager.init_track(box, appearance)
            new_track_ids.append(track_id)
        return new_track_ids

    def track_frame(self, img, boxes, threshold=0.5):
        if len(boxes) == 0:
            self.track_manager.predict_tracks()
            return [], []

        valid_boxes = []
        valid_indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                valid_boxes.append(box)
                valid_indices.append(i)

        if len(valid_boxes) == 0:
            return [], []

        valid_boxes = torch.stack(valid_boxes)

        detection_features = self.extract_detection_features(img, valid_boxes)

        if len(self.track_manager.tracks) == 0:
            track_ids = []
            for i in range(len(valid_boxes)):
                box = valid_boxes[i].detach().cpu().numpy()
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                track_ids.append(track_id)
            return track_ids, detection_features

        track_features, active_track_ids = self.extract_track_features()

        if len(track_features) == 0:
            track_ids = []
            for i in range(len(valid_boxes)):
                box = valid_boxes[i].detach().cpu().numpy()
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                track_ids.append(track_id)
            return track_ids, detection_features

        assignment = self.forward(detection_features, track_features)

        matched_track_indices = self.update_tracks(valid_boxes, detection_features, assignment, threshold)

        assigned_track_ids = []
        unmatched_det_indices = []

        for det_idx, track_idx in enumerate(matched_track_indices):
            if track_idx >= 0 and track_idx < len(active_track_ids):
                track_id = active_track_ids[track_idx]
                box = valid_boxes[det_idx].detach().cpu().numpy()
                appearance = detection_features[det_idx].detach().cpu().numpy()
                self.track_manager.update_track(track_id, box, appearance)
                assigned_track_ids.append(track_id)
            else:
                unmatched_det_indices.append(det_idx)

        new_track_ids = self.init_new_tracks(valid_boxes, detection_features, unmatched_det_indices)
        assigned_track_ids.extend(new_track_ids)

        self.track_manager.remove_deleted_tracks()

        return assigned_track_ids, detection_features