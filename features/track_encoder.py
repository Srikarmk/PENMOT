import torch
import torch.nn as nn
import numpy as np
from collections import deque


class TrackEncoder(nn.Module):
    def __init__(self,
                 appearance_dim=512,
                 motion_dim=8,
                 hidden_dim=256,
                 output_dim=512,
                 num_layers=2,
                 track_history_len=10):
        super(TrackEncoder, self).__init__()

        self.appearance_dim = appearance_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.track_history_len = track_history_len

        input_dim = appearance_dim + motion_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

        print(f"[OK] TrackEncoder initialized:")
        print(f"  Input: appearance({appearance_dim}) + motion({motion_dim}) = {input_dim}")
        print(f"  LSTM: {input_dim} -> {hidden_dim} (x{num_layers} layers)")
        print(f"  Output: {output_dim}D track features")
        print(f"  History length: {track_history_len} frames")

    def forward(self, track_histories):
        if len(track_histories) == 0:
            return torch.zeros((0, self.output_dim), device=next(self.parameters()).device)

        device = next(self.parameters()).device

        sequences = []
        seq_lengths = []

        for history in track_histories:
            recent_history = history[-self.track_history_len:]

            frames = []
            for frame_data in recent_history:
                appearance = frame_data['appearance']
                motion = frame_data['motion']

                if not isinstance(appearance, torch.Tensor):
                    appearance = torch.tensor(appearance, dtype=torch.float32, device=device)
                else:
                    appearance = appearance.to(device)

                if not isinstance(motion, torch.Tensor):
                    motion = torch.tensor(motion, dtype=torch.float32, device=device)
                else:
                    motion = motion.to(device)

                combined = torch.cat([appearance, motion])
                frames.append(combined)

            sequence = torch.stack(frames)
            sequences.append(sequence)
            seq_lengths.append(len(recent_history))

        max_len = max(seq_lengths)
        padded_sequences = []

        for seq, length in zip(sequences, seq_lengths):
            if length < max_len:
                padding = torch.zeros((max_len - length, seq.shape[1]), device=device)
                seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(seq)

        batch = torch.stack(padded_sequences).to(device)

        packed = nn.utils.rnn.pack_padded_sequence(
            batch,
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        lstm_out, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]
        track_features = self.projection(final_hidden)
        track_features = nn.functional.normalize(track_features, p=2, dim=1)

        return track_features


class Track:
    def __init__(self, track_id, initial_bbox, initial_appearance):
        self.track_id = track_id
        self.history = deque(maxlen=50)

        from features.kalman_filter import KalmanFilter
        self.kalman = KalmanFilter(initial_bbox, track_id)

        motion_state = self.kalman.get_state_vector()
        self.history.append({
            'bbox': initial_bbox,
            'appearance': initial_appearance,
            'motion': motion_state
        })

        self.age = 0
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox

    def update(self, bbox, appearance):
        self.kalman.update(bbox)

        motion_state = self.kalman.get_state_vector()

        self.history.append({
            'bbox': bbox,
            'appearance': appearance,
            'motion': motion_state
        })

        self.time_since_update = 0
        self.hits += 1

    def get_history(self):
        return list(self.history)

    def is_confirmed(self, min_hits=3):
        return self.hits >= min_hits

    def is_deleted(self, max_age=30):
        return self.time_since_update > max_age


class TrackManager:
    def __init__(self, track_encoder):
        self.track_encoder = track_encoder
        self.tracks = {}
        self.next_id = 1

    def init_track(self, bbox, appearance):
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = Track(track_id, bbox, appearance)
        return track_id

    def update_track(self, track_id, bbox, appearance):
        if track_id in self.tracks:
            self.tracks[track_id].update(bbox, appearance)

    def predict_tracks(self):
        predictions = {}
        for track_id, track in self.tracks.items():
            predictions[track_id] = track.predict()
        return predictions

    def get_track_features(self, track_ids=None):
        if track_ids is None:
            track_ids = list(self.tracks.keys())

        valid_ids = [tid for tid in track_ids if tid in self.tracks]

        if len(valid_ids) == 0:
            device = next(self.track_encoder.parameters()).device
            return torch.zeros((0, self.track_encoder.output_dim), device=device), []

        histories = [self.tracks[tid].get_history() for tid in valid_ids]

        features = self.track_encoder(histories)

        return features, valid_ids

    def remove_deleted_tracks(self, max_age=30):
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.is_deleted(max_age):
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return to_remove