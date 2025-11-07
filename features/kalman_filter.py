import numpy as np


class KalmanFilter:
    """
    Kalman Filter for tracking bounding boxes.
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
        - (x, y): center of bbox
        - (w, h): width and height
        - (vx, vy, vw, vh): velocities
    """
    
    def __init__(self, bbox, track_id):
        """
        Initialize Kalman Filter with initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            track_id: Unique identifier for this track
        """
        self.track_id = track_id
        self.age = 0  # Number of frames since track started
        self.time_since_update = 0  # Frames since last observation
        self.hits = 0  # Number of successful observations
        
        # Convert bbox to center format [cx, cy, w, h]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # State covariance matrix (uncertainty in state)
        self.covariance = np.eye(8, dtype=np.float32)
        self.covariance[4:, 4:] *= 10.0  # Higher uncertainty in velocities initially
        
        # Process noise (how much state can change between frames)
        self.process_noise = np.eye(8, dtype=np.float32)
        self.process_noise[4:, 4:] *= 0.01  # Small velocity changes
        
        # Measurement noise (uncertainty in observations)
        self.measurement_noise = np.eye(4, dtype=np.float32) * 1.0
        
        # State transition matrix (how state evolves)
        # Next state = current state + velocity
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1  # x = x + vx
        self.F[1, 5] = 1  # y = y + vy
        self.F[2, 6] = 1  # w = w + vw
        self.F[3, 7] = 1  # h = h + vh
        
        # Measurement matrix (we observe position, not velocity)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1  # Measure x
        self.H[1, 1] = 1  # Measure y
        self.H[2, 2] = 1  # Measure w
        self.H[3, 3] = 1  # Measure h
    
    def predict(self):
        """
        Predict next state (where object will be next frame).
        
        Returns:
            predicted_bbox: [x1, y1, x2, y2]
        """
        # Predict state: x_pred = F * x
        self.state = self.F @ self.state
        
        # Predict covariance: P_pred = F * P * F^T + Q
        self.covariance = self.F @ self.covariance @ self.F.T + self.process_noise
        
        self.age += 1
        self.time_since_update += 1
        
        # Convert to bbox format
        return self.get_bbox()
    
    def update(self, bbox):
        """
        Update state with new observation.
        
        Args:
            bbox: [x1, y1, x2, y2] observed bounding box
        """
        # Convert to center format
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        
        # Innovation (difference between observation and prediction)
        predicted_measurement = self.H @ self.state
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        S = self.H @ self.covariance @ self.H.T + self.measurement_noise
        
        # Kalman gain (how much to trust observation vs prediction)
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * innovation
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K*H) * P
        I = np.eye(8, dtype=np.float32)
        self.covariance = (I - K @ self.H) @ self.covariance
        
        self.time_since_update = 0
        self.hits += 1
    
    def get_bbox(self):
        """
        Get current bounding box in [x1, y1, x2, y2] format.
        """
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def get_state_vector(self):
        """Get full state including velocities."""
        return self.state.copy()
    
    def get_velocity(self):
        """Get velocity vector [vx, vy, vw, vh]."""
        return self.state[4:].copy()
    
    def is_confirmed(self, min_hits=3):
        """Check if track is confirmed (seen enough times)."""
        return self.hits >= min_hits
    
    def is_deleted(self, max_age=30):
        """Check if track should be deleted (not seen for too long)."""
        return self.time_since_update > max_age


class KalmanTracker:
    """
    Manages multiple Kalman Filters for tracking multiple objects.
    """
    
    def __init__(self):
        self.tracks = {}  # track_id -> KalmanFilter
        self.next_id = 1
    
    def init_track(self, bbox):
        """
        Initialize a new track.
        
        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            track_id: ID of new track
        """
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = KalmanFilter(bbox, track_id)
        return track_id
    
    def predict(self, track_id):
        """Predict next position for a track."""
        if track_id in self.tracks:
            return self.tracks[track_id].predict()
        return None
    
    def update(self, track_id, bbox):
        """Update track with new observation."""
        if track_id in self.tracks:
            self.tracks[track_id].update(bbox)
    
    def get_track(self, track_id):
        """Get track by ID."""
        return self.tracks.get(track_id, None)
    
    def get_all_predictions(self):
        """Get predictions for all tracks."""
        predictions = {}
        for track_id, track in self.tracks.items():
            predictions[track_id] = track.predict()
        return predictions
    
    def remove_deleted_tracks(self, max_age=30):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.is_deleted(max_age):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return to_remove