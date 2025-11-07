import torch
import torch.nn as nn
import numpy as np
from collections import deque


class TrackEncoder(nn.Module):
    """
    Encode track history using LSTM.
    
    Combines:
        1. Appearance features (from DetectionEncoder)
        2. Motion features (from KalmanFilter)
        3. Temporal patterns (LSTM)
    """
    
    def __init__(self, 
                 appearance_dim=512,
                 motion_dim=8,
                 hidden_dim=256,
                 output_dim=512,
                 num_layers=2,
                 track_history_len=10):
        """
        Args:
            appearance_dim: Dimension of appearance features from ResNet
            motion_dim: Dimension of motion state [x,y,w,h,vx,vy,vw,vh]
            hidden_dim: LSTM hidden dimension
            output_dim: Final track feature dimension
            num_layers: Number of LSTM layers
            track_history_len: Number of past frames to remember
        """
        super(TrackEncoder, self).__init__()
        
        self.appearance_dim = appearance_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.track_history_len = track_history_len
        
        # Input: concatenated appearance + motion features
        input_dim = appearance_dim + motion_dim
        
        # LSTM to encode temporal sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Project LSTM output to final feature dimension
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
        """
        Encode track histories.
        
        Args:
            track_histories: List of M track histories
                Each history is a list of dicts with keys:
                    - 'appearance': [appearance_dim] appearance feature
                    - 'motion': [motion_dim] motion state
                    
        Returns:
            track_features: [M, output_dim] encoded track features
        """
        if len(track_histories) == 0:
            return torch.zeros((0, self.output_dim), device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        
        # Convert histories to sequences
        sequences = []
        seq_lengths = []
        
        for history in track_histories:
            # Get recent history (up to track_history_len frames)
            recent_history = history[-self.track_history_len:]
            
            # Concatenate appearance + motion for each frame
            frames = []
            for frame_data in recent_history:
                appearance = frame_data['appearance']
                motion = frame_data['motion']
                
                # Ensure tensors
                if not isinstance(appearance, torch.Tensor):
                    appearance = torch.tensor(appearance, dtype=torch.float32)
                if not isinstance(motion, torch.Tensor):
                    motion = torch.tensor(motion, dtype=torch.float32)
                
                combined = torch.cat([appearance, motion])
                frames.append(combined)
            
            # Stack into sequence [seq_len, input_dim]
            sequence = torch.stack(frames)
            sequences.append(sequence)
            seq_lengths.append(len(recent_history))
        
        # Pad sequences to same length
        max_len = max(seq_lengths)
        padded_sequences = []
        
        for seq, length in zip(sequences, seq_lengths):
            if length < max_len:
                padding = torch.zeros((max_len - length, seq.shape[1]), device=device)
                seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(seq)
        
        # Stack into batch [M, max_len, input_dim]
        batch = torch.stack(padded_sequences).to(device)
        
        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            batch, 
            seq_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(packed)
        
        # Use final hidden state from last layer
        # hidden: [num_layers, M, hidden_dim]
        final_hidden = hidden[-1]  # [M, hidden_dim]
        
        # Project to output dimension
        track_features = self.projection(final_hidden)  # [M, output_dim]
        
        # L2 normalize
        track_features = nn.functional.normalize(track_features, p=2, dim=1)
        
        return track_features


class Track:
    """
    Represents a single object track with history.
    """
    
    def __init__(self, track_id, initial_bbox, initial_appearance):
        """
        Args:
            track_id: Unique track identifier
            initial_bbox: [x1, y1, x2, y2] initial bounding box
            initial_appearance: [appearance_dim] initial appearance feature
        """
        self.track_id = track_id
        self.history = deque(maxlen=50)  # Keep last 50 frames
        
        # Initialize Kalman Filter
        from features.kalman_filter import KalmanFilter
        self.kalman = KalmanFilter(initial_bbox, track_id)
        
        # Add initial observation
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
        """Predict next position."""
        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox
    
    def update(self, bbox, appearance):
        """
        Update track with new observation.
        
        Args:
            bbox: [x1, y1, x2, y2] new bounding box
            appearance: [appearance_dim] new appearance feature
        """
        # Update Kalman filter
        self.kalman.update(bbox)
        
        # Get updated motion state
        motion_state = self.kalman.get_state_vector()
        
        # Add to history
        self.history.append({
            'bbox': bbox,
            'appearance': appearance,
            'motion': motion_state
        })
        
        self.time_since_update = 0
        self.hits += 1
    
    def get_history(self):
        """Get track history as list."""
        return list(self.history)
    
    def is_confirmed(self, min_hits=3):
        """Check if track is confirmed."""
        return self.hits >= min_hits
    
    def is_deleted(self, max_age=30):
        """Check if track should be deleted."""
        return self.time_since_update > max_age


class TrackManager:
    """
    Manages multiple tracks and provides track features.
    """
    
    def __init__(self, track_encoder):
        """
        Args:
            track_encoder: TrackEncoder instance
        """
        self.track_encoder = track_encoder
        self.tracks = {}  # track_id -> Track
        self.next_id = 1
    
    def init_track(self, bbox, appearance):
        """
        Initialize a new track.
        
        Args:
            bbox: [x1, y1, x2, y2]
            appearance: [appearance_dim] appearance feature
            
        Returns:
            track_id: ID of new track
        """
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = Track(track_id, bbox, appearance)
        return track_id
    
    def update_track(self, track_id, bbox, appearance):
        """Update existing track."""
        if track_id in self.tracks:
            self.tracks[track_id].update(bbox, appearance)
    
    def predict_tracks(self):
        """Predict all tracks for next frame."""
        predictions = {}
        for track_id, track in self.tracks.items():
            predictions[track_id] = track.predict()
        return predictions
    
    def get_track_features(self, track_ids=None):
        """
        Get encoded features for tracks.
        
        Args:
            track_ids: List of track IDs (if None, use all tracks)
            
        Returns:
            features: [M, output_dim] track features
            track_ids: List of corresponding track IDs
        """
        if track_ids is None:
            track_ids = list(self.tracks.keys())
        
        # Filter confirmed tracks only
        confirmed_ids = [tid for tid in track_ids 
                        if tid in self.tracks and self.tracks[tid].is_confirmed()]
        
        if len(confirmed_ids) == 0:
            device = next(self.track_encoder.parameters()).device
            return torch.zeros((0, self.track_encoder.output_dim), device=device), []
        
        # Get histories
        histories = [self.tracks[tid].get_history() for tid in confirmed_ids]
        
        # Encode with LSTM
        features = self.track_encoder(histories)
        
        return features, confirmed_ids
    
    def remove_deleted_tracks(self, max_age=30):
        """Remove old tracks."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.is_deleted(max_age):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return to_remove