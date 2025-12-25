"""
bytetrack.py - Enhanced ByteTrack with better ID persistence
Improvements:
1. Increased track buffer for longer occlusions
2. More lenient matching after occlusion
3. Appearance features (simple color histogram)
4. Better Kalman prediction
"""
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy import linalg
import cv2


class KalmanFilter:
    """Improved Kalman Filter with better motion model"""
    
    def __init__(self):
        self.ndim = 4
        self.dt = 1.0
        
        # State transition matrix
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Observation matrix
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Reduced noise for more stable predictions during occlusion
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """Initialize state from first detection"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[2],
            1e-2,
            2 * self._std_weight_position * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            1e-5,
            10 * self._std_weight_velocity * measurement[2]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Predict next state"""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-2,
            self._std_weight_position * mean[2]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[2],
            1e-5,
            self._std_weight_velocity * mean[2]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state to measurement space"""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-1,
            self._std_weight_position * mean[2]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Update state with measurement"""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        try:
            chol_factor, lower = linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T
        except:
            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.inv(projected_cov)
            )
        
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class STrack:
    """Enhanced single target track with appearance features"""
    
    shared_kalman = KalmanFilter()
    track_id = 0
    
    def __init__(self, tlwh, score, class_id, frame=None):
        """Initialize track with optional appearance feature"""
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.class_id = class_id
        self.tracklet_len = 0
        
        self.state = 'new'
        self.frame_id = 0
        self.start_frame = 0
        
        # Store appearance feature (simple color histogram)
        self.appearance_feature = None
        if frame is not None:
            self._extract_appearance(frame, tlwh)
        
        # Track history for smoother predictions
        self.history = deque(maxlen=30)
        
    def _extract_appearance(self, frame, tlwh):
        """Extract simple color histogram as appearance feature"""
        try:
            x, y, w, h = tlwh.astype(int)
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            if x2 > x and y2 > y:
                crop = frame[y:y2, x:x2]
                if crop.size > 0:
                    # Compute HSV histogram
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
                    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                    
                    # Normalize
                    hist_h = cv2.normalize(hist_h, hist_h).flatten()
                    hist_s = cv2.normalize(hist_s, hist_s).flatten()
                    
                    self.appearance_feature = np.concatenate([hist_h, hist_s])
        except:
            self.appearance_feature = None
    
    def activate(self, frame_id):
        """Activate new track"""
        STrack.track_id += 1
        self.track_id = STrack.track_id
        
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate lost track with new detection"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.class_id = new_track.class_id
        
        # Update appearance feature
        if new_track.appearance_feature is not None:
            if self.appearance_feature is not None:
                # Smooth update: 80% old, 20% new
                self.appearance_feature = 0.8 * self.appearance_feature + 0.2 * new_track.appearance_feature
            else:
                self.appearance_feature = new_track.appearance_feature
        
        if new_id:
            STrack.track_id += 1
            self.track_id = STrack.track_id
    
    def update(self, new_track, frame_id):
        """Update matched track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = 'tracked'
        self.is_activated = True
        
        self.score = new_track.score
        self.class_id = new_track.class_id
        
        # Update appearance feature smoothly
        if new_track.appearance_feature is not None:
            if self.appearance_feature is not None:
                self.appearance_feature = 0.8 * self.appearance_feature + 0.2 * new_track.appearance_feature
            else:
                self.appearance_feature = new_track.appearance_feature
    
    def predict(self):
        """Predict next state"""
        mean_state = self.mean.copy()
        if self.state != 'tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)
        
        # Store prediction in history
        self.history.append(self.tlbr)
    
    @property
    def tlwh(self):
        """Get current position in tlwh format"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Get current position in tlbr format"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def mark_lost(self):
        self.state = 'lost'
    
    def mark_removed(self):
        self.state = 'removed'


class ByteTracker:
    """Enhanced ByteTrack with better ID persistence"""
    
    def __init__(self, track_thresh=0.5, track_buffer=50, match_thresh=0.8, 
                 use_appearance=True, appearance_thresh=0.25):
        """
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Frames to keep lost tracks (increased from 30)
            match_thresh: IOU threshold for matching
            use_appearance: Whether to use appearance features
            appearance_thresh: Appearance matching threshold
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.use_appearance = use_appearance
        self.appearance_thresh = appearance_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.current_frame = None
        STrack.track_id = 0
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections: numpy array [[x1, y1, x2, y2, conf, class_id], ...]
            frame: Original frame for appearance features (optional)
            
        Returns:
            tracks: [[x1, y1, x2, y2, track_id, class_id, conf], ...]
        """
        self.frame_id += 1
        self.current_frame = frame
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(detections) > 0:
            # Split detections
            high_dets = detections[detections[:, 4] >= self.track_thresh]
            low_dets = detections[detections[:, 4] < self.track_thresh]
            
            # Convert to STrack objects with appearance
            detections_high = [
                STrack(self._xyxy_to_tlwh(det[:4]), det[4], int(det[5]), frame)
                for det in high_dets
            ]
            detections_low = [
                STrack(self._xyxy_to_tlwh(det[:4]), det[4], int(det[5]), frame)
                for det in low_dets
            ]
        else:
            detections_high = []
            detections_low = []
        
        # Separate tracked and unconfirmed
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Predict all tracks
        strack_pool = self._joint_tracks(tracked_stracks, self.lost_stracks)
        for strack in strack_pool:
            strack.predict()
        
        # ========== First association: High confidence ==========
        matches, u_track, u_detection = self._associate(
            strack_pool, detections_high, self.match_thresh
        )
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ========== Second association: Low confidence ==========
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 'tracked']
        
        # Use more lenient threshold for low-confidence matching
        matches_low, u_track_low, u_detection_low = self._associate(
            r_tracked_stracks, detections_low, 0.5
        )
        
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # ========== Third association: Lost tracks with appearance ==========
        # Try to match remaining lost tracks using appearance + relaxed IOU
        if self.use_appearance and len(u_track_low) > 0 and len(u_detection) > 0:
            lost_tracks_remain = [r_tracked_stracks[i] for i in u_track_low if r_tracked_stracks[i].state == 'tracked']
            dets_remain = [detections_high[i] for i in u_detection]
            
            if len(lost_tracks_remain) > 0 and len(dets_remain) > 0:
                matches_app, u_track_app, u_detection_app = self._associate_with_appearance(
                    lost_tracks_remain, dets_remain
                )
                
                for itracked, idet in matches_app:
                    track = lost_tracks_remain[itracked]
                    det = dets_remain[idet]
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                
                # Update unmatched indices
                u_track_final = [u_track_low[i] for i in u_track_app if i < len(u_track_low)]
                u_detection = [u_detection[i] for i in u_detection_app if i < len(u_detection)]
            else:
                u_track_final = u_track_low
        else:
            u_track_final = u_track_low
        
        # Mark remaining unmatched tracks as lost
        for it in u_track_final:
            track = r_tracked_stracks[it]
            if track.state != 'lost':
                track.mark_lost()
                lost_stracks.append(track)
        
        # ========== Init new tracks ==========
        for inew in u_detection:
            track = detections_high[inew]
            if track.score >= self.track_thresh:
                track.activate(self.frame_id)
                activated_stracks.append(track)
        
        # ========== Update states ==========
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks = self._joint_tracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._joint_tracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self._sub_tracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_tracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # Get output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        results = []
        for track in output_stracks:
            tlbr = track.tlbr
            results.append([
                tlbr[0], tlbr[1], tlbr[2], tlbr[3],
                track.track_id, track.class_id, track.score
            ])
        
        return np.array(results) if results else np.empty((0, 7))
    
    def _associate(self, tracks, detections, thresh):
        """Associate using IOU"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(tracks))), list(range(len(detections)))
        
        iou_matrix = self._iou_distance(tracks, detections)
        matches, u_track, u_detection = self._linear_assignment(iou_matrix, thresh)
        
        return matches, u_track, u_detection
    
    def _associate_with_appearance(self, tracks, detections):
        """Associate using combined IOU + appearance similarity"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IOU distance (relaxed threshold)
        iou_dist = self._iou_distance(tracks, detections)
        
        # Compute appearance distance
        app_dist = self._appearance_distance(tracks, detections)
        
        # Combined distance: 50% IOU + 50% appearance
        if app_dist is not None:
            combined_dist = 0.5 * iou_dist + 0.5 * app_dist
        else:
            combined_dist = iou_dist
        
        # Use more lenient threshold for lost track recovery
        matches, u_track, u_detection = self._linear_assignment(combined_dist, 0.6)
        
        return matches, u_track, u_detection
    
    def _appearance_distance(self, tracks, detections):
        """Compute appearance distance using color histograms"""
        try:
            dist_matrix = np.zeros((len(tracks), len(detections)))
            
            for i, track in enumerate(tracks):
                for j, det in enumerate(detections):
                    if track.appearance_feature is not None and det.appearance_feature is not None:
                        # Compute cosine distance
                        similarity = np.dot(track.appearance_feature, det.appearance_feature)
                        similarity /= (np.linalg.norm(track.appearance_feature) * 
                                     np.linalg.norm(det.appearance_feature) + 1e-6)
                        dist_matrix[i, j] = 1.0 - similarity
                    else:
                        dist_matrix[i, j] = 1.0  # Maximum distance if no feature
            
            return dist_matrix
        except:
            return None
    
    def _iou_distance(self, atracks, btracks):
        """Compute IOU distance"""
        atlbrs = np.array([track.tlbr for track in atracks])
        btlbrs = np.array([track.tlbr for track in btracks])
        
        ious = self._batch_iou(atlbrs, btlbrs)
        return 1 - ious
    
    @staticmethod
    def _batch_iou(bboxes1, bboxes2):
        """Compute IOU between two sets of boxes"""
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)
        
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        
        return wh / (area1 + area2 - wh + 1e-6)
    
    @staticmethod
    def _linear_assignment(cost_matrix, thresh):
        """Linear assignment with threshold"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        try:
            import lap
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        except ImportError:
            cost_matrix_thresh = cost_matrix.copy()
            cost_matrix_thresh[cost_matrix_thresh > thresh] = thresh + 1e5
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix_thresh)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= thresh:
                    matches.append([r, c])
            
            matched_rows = set([m[0] for m in matches])
            matched_cols = set([m[1] for m in matches])
            
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
            unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]
            
            unmatched_a = np.array(unmatched_a)
            unmatched_b = np.array(unmatched_b)
        
        matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
        
        return matches, unmatched_a, unmatched_b
    
    @staticmethod
    def _xyxy_to_tlwh(bbox):
        """Convert xyxy to tlwh"""
        return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    @staticmethod
    def _joint_tracks(tlista, tlistb):
        """Join two track lists"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res
    
    @staticmethod
    def _sub_tracks(tlista, tlistb):
        """Subtract track list b from a"""
        tracks = {}
        for t in tlista:
            tracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if tracks.get(tid, 0):
                del tracks[tid]
        return list(tracks.values())
