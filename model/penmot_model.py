# import torch
# import torch.nn as nn
# import sys
# import os
#
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)
#
# from features.detection_encoder import DetectionEncoder
# from features.track_encoder import TrackEncoder, TrackManager
# from model.set_transformer import SetTransformer
# from model.sinkhorn import SinkhornMatcher
#
#
# class PENMOT(nn.Module):
#     def __init__(self,
#                  detection_feature_dim=512,
#                  track_feature_dim=512,
#                  transformer_dim=512,
#                  num_heads=8,
#                  num_layers=3,
#                  sinkhorn_iters=5,
#                  sinkhorn_tau=0.1,
#                  freeze_detection_encoder=False):
#         super().__init__()
#
#         self.detection_encoder = DetectionEncoder(
#             feature_dim=detection_feature_dim,
#             pretrained=True,
#             freeze_backbone=freeze_detection_encoder
#         )
#
#         self.track_encoder = TrackEncoder(
#             appearance_dim=detection_feature_dim,
#             motion_dim=8,
#             hidden_dim=256,
#             output_dim=track_feature_dim
#         )
#
#         self.detection_projection = nn.Linear(detection_feature_dim, transformer_dim)
#         self.track_projection = nn.Linear(track_feature_dim, transformer_dim)
#
#         self.detection_transformer = SetTransformer(
#             d_model=transformer_dim,
#             num_heads=num_heads,
#             num_layers=num_layers
#         )
#
#         self.track_transformer = SetTransformer(
#             d_model=transformer_dim,
#             num_heads=num_heads,
#             num_layers=num_layers
#         )
#
#         self.matcher = SinkhornMatcher(
#             n_iters=sinkhorn_iters,
#             tau=sinkhorn_tau
#         )
#
#         self.track_manager = TrackManager(self.track_encoder)
#
#         print(f"[OK] PENMOT initialized")
#
#     def extract_detection_features(self, img, boxes):
#         if len(boxes) == 0:
#             return torch.zeros((0, 512), device=img.device)
#
#         valid_boxes = []
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             if x2 > x1 + 10 and y2 > y1 + 10:
#                 valid_boxes.append(box)
#
#         if len(valid_boxes) == 0:
#             return torch.zeros((0, 512), device=img.device)
#
#         valid_boxes = torch.stack(valid_boxes)
#         return self.detection_encoder(img, valid_boxes)
#
#     def extract_track_features(self, track_ids=None):
#         return self.track_manager.get_track_features(track_ids)
#
#     def forward(self, detection_features, track_features):
#         if len(detection_features) == 0 or len(track_features) == 0:
#             return torch.zeros((len(detection_features), len(track_features)))
#
#         det_proj = self.detection_projection(detection_features)
#         track_proj = self.track_projection(track_features)
#
#         det_out = self.detection_transformer(det_proj, track_proj)
#         track_out = self.track_transformer(track_proj, det_proj)
#
#         cost_matrix = -torch.mm(det_out, track_out.t())
#
#         assignment = self.matcher(cost_matrix)
#
#         return assignment
#
#     def update_tracks(self, boxes, detection_features, assignment, threshold=0.5):
#         matched_tracks = []
#
#         for det_idx in range(len(boxes)):
#             if len(assignment.shape) == 2:
#                 track_idx = torch.argmax(assignment[det_idx])
#                 confidence = assignment[det_idx, track_idx]
#             else:
#                 track_idx = torch.argmax(assignment)
#                 confidence = assignment[track_idx]
#
#             if confidence > threshold:
#                 matched_tracks.append(track_idx.item())
#             else:
#                 matched_tracks.append(-1)
#
#         return matched_tracks
#
#     def init_new_tracks(self, boxes, detection_features, unmatched_indices):
#         new_track_ids = []
#         for idx in unmatched_indices:
#             box = boxes[idx].detach().cpu().numpy()
#             appearance = detection_features[idx].detach().cpu().numpy()
#             track_id = self.track_manager.init_track(box, appearance)
#             new_track_ids.append(track_id)
#         return new_track_ids
#
#     def track_frame(self, img, boxes, threshold=0.5):
#         if len(boxes) == 0:
#             self.track_manager.predict_tracks()
#             return [], []
#
#         valid_boxes = []
#         valid_indices = []
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box
#             if x2 > x1 and y2 > y1:
#                 valid_boxes.append(box)
#                 valid_indices.append(i)
#
#         if len(valid_boxes) == 0:
#             return [], []
#
#         valid_boxes = torch.stack(valid_boxes)
#
#         detection_features = self.extract_detection_features(img, valid_boxes)
#
#         if len(self.track_manager.tracks) == 0:
#             track_ids = []
#             for i in range(len(valid_boxes)):
#                 box = valid_boxes[i].detach().cpu().numpy()
#                 appearance = detection_features[i].detach().cpu().numpy()
#                 track_id = self.track_manager.init_track(box, appearance)
#                 track_ids.append(track_id)
#             return track_ids, detection_features
#
#         track_features, active_track_ids = self.extract_track_features()
#
#         if len(track_features) == 0:
#             track_ids = []
#             for i in range(len(valid_boxes)):
#                 box = valid_boxes[i].detach().cpu().numpy()
#                 appearance = detection_features[i].detach().cpu().numpy()
#                 track_id = self.track_manager.init_track(box, appearance)
#                 track_ids.append(track_id)
#             return track_ids, detection_features
#
#         assignment = self.forward(detection_features, track_features)
#
#         matched_track_indices = self.update_tracks(valid_boxes, detection_features, assignment, threshold)
#
#         assigned_track_ids = []
#         unmatched_det_indices = []
#
#         for det_idx, track_idx in enumerate(matched_track_indices):
#             if track_idx >= 0 and track_idx < len(active_track_ids):
#                 track_id = active_track_ids[track_idx]
#                 box = valid_boxes[det_idx].detach().cpu().numpy()
#                 appearance = detection_features[det_idx].detach().cpu().numpy()
#                 self.track_manager.update_track(track_id, box, appearance)
#                 assigned_track_ids.append(track_id)
#             else:
#                 unmatched_det_indices.append(det_idx)
#
#         new_track_ids = self.init_new_tracks(valid_boxes, detection_features, unmatched_det_indices)
#         assigned_track_ids.extend(new_track_ids)
#
#         self.track_manager.remove_deleted_tracks()
#
#         return assigned_track_ids, detection_features

# import torch
# import torch.nn as nn
# import sys
# import os

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# from features.detection_encoder import DetectionEncoder
# from features.track_encoder import TrackEncoder, TrackManager
# from model.set_transformer import SetTransformer
# from model.sinkhorn import SinkhornMatcher


# class PENMOT(nn.Module):
#     def __init__(self,
#                  detection_feature_dim=512,
#                  track_feature_dim=512,
#                  transformer_dim=512,
#                  num_heads=8,
#                  num_layers=3,
#                  sinkhorn_iters=5,
#                  sinkhorn_tau=0.1,
#                  freeze_detection_encoder=False):
#         super().__init__()

#         self.detection_encoder = DetectionEncoder(
#             feature_dim=detection_feature_dim,
#             pretrained=True,
#             freeze_backbone=freeze_detection_encoder
#         )

#         self.track_encoder = TrackEncoder(
#             appearance_dim=detection_feature_dim,
#             motion_dim=8,
#             hidden_dim=256,
#             output_dim=track_feature_dim
#         )

#         self.detection_projection = nn.Linear(detection_feature_dim, transformer_dim)
#         self.track_projection = nn.Linear(track_feature_dim, transformer_dim)

#         self.detection_transformer = SetTransformer(
#             d_model=transformer_dim,
#             num_heads=num_heads,
#             num_layers=num_layers
#         )

#         self.track_transformer = SetTransformer(
#             d_model=transformer_dim,
#             num_heads=num_heads,
#             num_layers=num_layers
#         )

#         self.matcher = SinkhornMatcher(
#             n_iters=sinkhorn_iters,
#             tau=sinkhorn_tau
#         )

#         self.track_manager = TrackManager(self.track_encoder)

#         print(f"[OK] PENMOT initialized")

#     def extract_detection_features(self, img, boxes):
#         if len(boxes) == 0:
#             return torch.zeros((0, 512), device=img.device)

#         valid_boxes = []
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             if x2 > x1 + 10 and y2 > y1 + 10:
#                 valid_boxes.append(box)

#         if len(valid_boxes) == 0:
#             return torch.zeros((0, 512), device=img.device)

#         valid_boxes = torch.stack(valid_boxes)
#         return self.detection_encoder(img, valid_boxes)

#     def extract_track_features(self, track_ids=None):
#         return self.track_manager.get_track_features(track_ids)

#     def forward(self, detection_features, track_features):
#         if len(detection_features) == 0 or len(track_features) == 0:
#             return torch.zeros((len(detection_features), len(track_features)))

#         det_proj = self.detection_projection(detection_features)
#         track_proj = self.track_projection(track_features)

#         det_out = self.detection_transformer(det_proj, track_proj)
#         track_out = self.track_transformer(track_proj, det_proj)

#         cost_matrix = -torch.mm(det_out, track_out.t())

#         assignment = self.matcher(cost_matrix)

#         return assignment

#     def update_tracks(self, boxes, detection_features, assignment, threshold=0.5):
#         matched_tracks = []

#         for det_idx in range(len(boxes)):
#             if len(assignment.shape) == 2:
#                 track_idx = torch.argmax(assignment[det_idx])
#                 confidence = assignment[det_idx, track_idx]
#             else:
#                 track_idx = torch.argmax(assignment)
#                 confidence = assignment[track_idx]

#             if confidence > threshold:
#                 matched_tracks.append(track_idx.item())
#             else:
#                 matched_tracks.append(-1)

#         return matched_tracks

#     def init_new_tracks(self, boxes, detection_features, unmatched_indices):
#         new_track_ids = []
#         for idx in unmatched_indices:
#             box = boxes[idx].detach().cpu().numpy()
#             appearance = detection_features[idx].detach().cpu().numpy()
#             track_id = self.track_manager.init_track(box, appearance)
#             new_track_ids.append(track_id)
#         return new_track_ids

#     def track_frame(self, img, boxes, threshold=0.5):
#         if len(boxes) == 0:
#             self.track_manager.predict_tracks()
#             return [], []

#         valid_boxes = []
#         valid_indices = []
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box
#             if x2 > x1 and y2 > y1:
#                 valid_boxes.append(box)
#                 valid_indices.append(i)

#         if len(valid_boxes) == 0:
#             return [], []

#         valid_boxes = torch.stack(valid_boxes)

#         detection_features = self.extract_detection_features(img, valid_boxes)

#         if len(self.track_manager.tracks) == 0:
#             track_ids = []
#             for i in range(len(valid_boxes)):
#                 box = valid_boxes[i].detach().cpu().numpy()
#                 appearance = detection_features[i].detach().cpu().numpy()
#                 track_id = self.track_manager.init_track(box, appearance)
#                 track_ids.append(track_id)
#             return track_ids, detection_features

#         track_features, active_track_ids = self.extract_track_features()

#         if len(track_features) == 0:
#             track_ids = []
#             for i in range(len(valid_boxes)):
#                 box = valid_boxes[i].detach().cpu().numpy()
#                 appearance = detection_features[i].detach().cpu().numpy()
#                 track_id = self.track_manager.init_track(box, appearance)
#                 track_ids.append(track_id)
#             return track_ids, detection_features

#         assignment = self.forward(detection_features, track_features)

#         matched_track_indices = self.update_tracks(valid_boxes, detection_features, assignment, threshold)

#         assigned_track_ids = []
#         unmatched_det_indices = []

#         for det_idx, track_idx in enumerate(matched_track_indices):
#             if track_idx >= 0 and track_idx < len(active_track_ids):
#                 track_id = active_track_ids[track_idx]
#                 box = valid_boxes[det_idx].detach().cpu().numpy()
#                 appearance = detection_features[det_idx].detach().cpu().numpy()
#                 self.track_manager.update_track(track_id, box, appearance)
#                 assigned_track_ids.append(track_id)
#             else:
#                 unmatched_det_indices.append(det_idx)

#         new_track_ids = self.init_new_tracks(valid_boxes, detection_features, unmatched_det_indices)
#         assigned_track_ids.extend(new_track_ids)

#         self.track_manager.remove_deleted_tracks()

#         return assigned_track_ids, detection_features
import torch
import torch.nn as nn
import sys
import os
import numpy as np

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
            tau=sinkhorn_tau,
            hard_assignment=False
        )

        self.inference_matcher = SinkhornMatcher(
            n_iters=sinkhorn_iters,
            tau=sinkhorn_tau,
            hard_assignment=True
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

    def compute_motion_cost(self, detection_boxes, track_boxes):
        det_boxes_np = detection_boxes.cpu().numpy() if torch.is_tensor(detection_boxes) else detection_boxes
        track_boxes_np = track_boxes.cpu().numpy() if torch.is_tensor(track_boxes) else track_boxes

        motion_cost = np.zeros((len(det_boxes_np), len(track_boxes_np)))

        for i, det_box in enumerate(det_boxes_np):
            for j, track_box in enumerate(track_boxes_np):
                x1 = max(det_box[0], track_box[0])
                y1 = max(det_box[1], track_box[1])
                x2 = min(det_box[2], track_box[2])
                y2 = min(det_box[3], track_box[3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                area2 = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                union = area1 + area2 - inter

                iou = inter / (union + 1e-8)
                motion_cost[i, j] = iou

        return torch.tensor(motion_cost, dtype=torch.float32,
                            device=detection_boxes.device if torch.is_tensor(detection_boxes) else torch.device('cpu'))

    def forward(self, detection_features, track_features, detection_boxes=None, track_boxes=None,
                motion_weight=0.3, use_hard_assignment=None):
        if len(detection_features) == 0 or len(track_features) == 0:
            return torch.zeros((len(detection_features), len(track_features)))

        det_proj = self.detection_projection(detection_features)
        track_proj = self.track_projection(track_features)

        det_out = self.detection_transformer(det_proj, track_proj)
        track_out = self.track_transformer(track_proj, det_proj)

        appearance_similarity = torch.mm(det_out, track_out.t())

        if detection_boxes is not None and track_boxes is not None:
            motion_similarity = self.compute_motion_cost(detection_boxes, track_boxes)
            combined_similarity = (1 - motion_weight) * appearance_similarity + motion_weight * motion_similarity
            cost_matrix = -combined_similarity
        else:
            cost_matrix = -appearance_similarity

        if use_hard_assignment is None:
            use_hard_assignment = not self.training

        matcher = self.inference_matcher if use_hard_assignment else self.matcher
        assignment = matcher(cost_matrix)

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

    def track_frame(self, img, boxes, motion_weight=0.3):
        if len(boxes) == 0:
            self.track_manager.predict_tracks()
            return [], []

        valid_boxes = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                valid_boxes.append(box)

        if len(valid_boxes) == 0:
            return [], []

        valid_boxes = torch.stack(valid_boxes)
        valid_boxes_np = valid_boxes.detach().cpu().numpy()

        detection_features = self.extract_detection_features(img, valid_boxes)

        if len(self.track_manager.tracks) == 0:
            track_ids = []
            for i in range(len(valid_boxes)):
                box = valid_boxes_np[i]
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                track_ids.append(track_id)
            return track_ids, detection_features

        confirmed_tracks = {tid: t for tid, t in self.track_manager.tracks.items()
                            if t.is_confirmed(min_hits=1)}

        if len(confirmed_tracks) == 0:
            track_ids = []
            for i in range(len(valid_boxes)):
                box = valid_boxes_np[i]
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                track_ids.append(track_id)
            return track_ids, detection_features

        active_track_ids = list(confirmed_tracks.keys())

        track_boxes_list = []
        for tid in active_track_ids:
            predicted_box = self.track_manager.tracks[tid].kalman.get_bbox()
            track_boxes_list.append(predicted_box)

        track_boxes = torch.tensor(np.array(track_boxes_list), dtype=torch.float32, device=valid_boxes.device)

        track_features, feature_track_ids = self.extract_track_features(active_track_ids)

        if len(track_features) == 0:
            track_ids = []
            for i in range(len(valid_boxes)):
                box = valid_boxes_np[i]
                appearance = detection_features[i].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                track_ids.append(track_id)
            return track_ids, detection_features

        assignment = self.forward(detection_features, track_features,
                                  valid_boxes, track_boxes,
                                  motion_weight=motion_weight,
                                  use_hard_assignment=True)

        matches = (assignment > 0.5).nonzero(as_tuple=False)

        assigned_track_ids = [-1] * len(valid_boxes)

        for match in matches:
            det_idx = match[0].item()
            track_idx = match[1].item()

            if det_idx < len(valid_boxes) and track_idx < len(feature_track_ids):
                track_id = feature_track_ids[track_idx]
                assigned_track_ids[det_idx] = track_id

                box = valid_boxes_np[det_idx]
                appearance = detection_features[det_idx].detach().cpu().numpy()
                self.track_manager.update_track(track_id, box, appearance)

        for det_idx in range(len(valid_boxes)):
            if assigned_track_ids[det_idx] == -1:
                box = valid_boxes_np[det_idx]
                appearance = detection_features[det_idx].detach().cpu().numpy()
                track_id = self.track_manager.init_track(box, appearance)
                assigned_track_ids[det_idx] = track_id

        self.track_manager.remove_deleted_tracks(max_age=30)

        return assigned_track_ids, detection_features