#!/usr/bin/env python3
"""
/home/foxsec/drone_project/follow_target.py

Main program for autonomous drone single-target tracking with re-identification.
STEP 2 (Refined with R1 - R6 & PART D Experiments E1 - E3):
- R1: Pretrained Person ReID weights (OSNet MSMT17/Market1501) loaded from weights/
- R2: Tiered Quality Filter (FULL, PARTIAL, NONE) replacing hard frame border exclusion
- R3: Anchor Floor (sim_anchor >= 0.60), safe dynamic additions, and 60-frame rollback
- R4: Strict target appearance continuity check (< 0.50 consecutive sim -> immediate LOST)
- R5: Robust anchor collection (spacing >= 5 frames, cosine sim <= 0.95, continuity check)
- R6: Gallery debug crops in outputs/gallery_debug/, enriched CSV columns, and false-lock report
- E1: View chain update (USE_VIEW_CHAIN_UPDATE) for learning back-view features safely
- E2: HSV Color Cue toggle (USE_COLOR_CUE)
- E3: ReID model selection (OSNet, OSNet-AIN)
"""

import os
import sys
import time
import argparse
import warnings
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", message="Cython evaluation")
try:
    import torchreid
except ImportError:
    torchreid = None
from ultralytics import YOLO


# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

DEVICE = 0                           # GPU device ID
YOLO_MODEL_PATH = "yolov8n.pt"       # YOLOv8 nano detector
CONF_THRESHOLD = 0.40                # Detection confidence threshold
TARGET_CLASSES = [0]                 # 0 = person in COCO
TRACKER_CONFIG = "botsort.yaml"      # BoT-SORT tracker config

REID_MODEL_NAME = "osnet_x1_0"       # Default ReID model
REID_WEIGHTS_PATH = "/home/foxsec/drone_project/weights/osnet_x1_0_msmt17.pth"
REID_WEIGHTS_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
REID_INPUT_SIZE = (128, 256)
USE_COLOR_CUE = False                # Toggle via --use-color-cue (E2)
REID_SIM_WEIGHT = 0.80
COLOR_CUE_WEIGHT = 0.20

CLASS_QUALITY_CONFIG = {
    0: {
        "min_height": 80,
        "min_width": 30,
        "border_margin": 5,
        "max_overlap_iou": 0.30,
        "use_color_cue": False,
    }
}

ANCHOR_GALLERY_SIZE = 10
DYNAMIC_GALLERY_MAX_SIZE = 30
VIEW_GALLERY_MAX_SIZE = 30           # E1: Max size for view chain gallery
ANCHOR_FRAME_SPACING = 5
ANCHOR_MAX_SIM = 0.95
T_ANCHOR_FLOOR = 0.60                # Hard minimum similarity against anchor gallery
DYNAMIC_ADD_ANCHOR_MIN = 0.80
DYNAMIC_ADD_SUSPICIOUS_SPACING = 30
DYNAMIC_ROLLBACK_WINDOW = 60
T_UPDATE = 0.75
MAX_SIM_FOR_ADDITION = 0.95
MIN_FRAMES_BETWEEN_ADDITIONS = 5
TOP_K_GALLERY_SIM = 3
VELOCITY_SMOOTHING_WINDOW = 5

USE_SPATIAL_GATE = False
SPATIAL_GATE_TIME_LIMIT = 3.0
SPATIAL_GATE_R0 = 150.0
SPATIAL_GATE_SPEED = 200.0
T_ACCEPT_SHORT = 0.70
T_ACCEPT_LONG = 0.78
PARTIAL_ACCEPT_PENALTY = 0.05
N_CONFIRM = 8
MARGIN_MIN = 0.08
T_MAX_LOST = 30.0
MISSING_FRAMES_FOR_LOST = 3

T_VERIFY = 0.55
VERIFY_INTERVAL_FRAMES = 5
VERIFY_FAIL_LIMIT = 3
SUSPICIOUS_WINDOW_FRAMES = 15
SUSPICIOUS_FAIL_LIMIT = 2
OVERLAP_IOU_THRESHOLD = 0.30
CONSECUTIVE_EMB_SIM_THRESH = 0.50
CENTER_JUMP_RATIO_THRESH = 0.25
AREA_CHANGE_RATIO_THRESH = 2.0

# Experiment E1 Switch
USE_VIEW_CHAIN_UPDATE = False        # Toggle via --use-view-chain (E1)

GALLERY_DEBUG_DIR = "/home/foxsec/drone_project/outputs/gallery_debug"


# ==============================================================================
# DATA STRUCTURES & BASE CLASSES
# ==============================================================================

class TargetState(Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    GIVE_UP = "GIVE_UP"


class QualityTier(Enum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    FULL = "FULL"


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def compute_color_histogram(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return np.zeros((256,), dtype=np.float32)

    upper = crop[:h // 2, :]
    lower = crop[h // 2:, :]

    def _calc_hsv_hist(img_part: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        norm = np.linalg.norm(hist)
        return hist.flatten() / (norm + 1e-6)

    h_upper = _calc_hsv_hist(upper)
    h_lower = _calc_hsv_hist(lower)
    combined = np.concatenate([h_upper, h_lower])
    combined /= (np.linalg.norm(combined) + 1e-6)
    return combined.astype(np.float32)


class Track:
    def __init__(self, track_id: int, box: Tuple[int, int, int, int], conf: float, cls_id: int):
        self.track_id = int(track_id)
        self.box = tuple(int(v) for v in box)
        self.conf = float(conf)
        self.cls_id = int(cls_id)

    @property
    def x1(self) -> int: return self.box[0]
    @property
    def y1(self) -> int: return self.box[1]
    @property
    def x2(self) -> int: return self.box[2]
    @property
    def y2(self) -> int: return self.box[3]
    @property
    def width(self) -> int: return max(0, self.x2 - self.x1)
    @property
    def height(self) -> int: return max(0, self.y2 - self.y1)
    @property
    def area(self) -> int: return self.width * self.height
    @property
    def center(self) -> Tuple[float, float]: return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, pt: Tuple[int, int]) -> bool:
        px, py = pt
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


# ==============================================================================
# FEATURE EXTRACTOR & QUALITY FILTER
# ==============================================================================

class FeatureExtractor(ABC):
    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray: pass


class PersonReIDExtractor(FeatureExtractor):
    def __init__(self,
                 model_name: str = REID_MODEL_NAME,
                 weights_path: str = REID_WEIGHTS_PATH,
                 device: Any = DEVICE):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        print(f"[PersonReIDExtractor] Initializing {model_name} on {self.device}...")

        if torchreid is not None:
            self.model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=False)
        else:
            import importlib.util
            osnet_path = "/home/foxsec/drone_project/venv/lib/python3.12/site-packages/torchreid/reid/models/osnet.py"
            spec = importlib.util.spec_from_file_location("osnet_module", osnet_path)
            osnet_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(osnet_mod)
            factory_fn = getattr(osnet_mod, model_name)
            self.model = factory_fn(num_classes=1000, pretrained=False)

        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(os.path.abspath(weights_path)), exist_ok=True)
            print(f"[PersonReIDExtractor] Weights not found at {weights_path}, downloading...")
            import urllib.request
            urllib.request.urlretrieve(REID_WEIGHTS_URL, weights_path)

        weight_filename = os.path.basename(weights_path)
        print(f"[PersonReIDExtractor] Loading pretrained ReID weights: {weight_filename}")

        if torchreid is not None:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model_dict = self.model.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            self.model.load_state_dict(model_dict)

        self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, REID_INPUT_SIZE)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor

    def embed(self, crop: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(crop).to(self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat / torch.norm(feat, p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]

    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        tensors = []
        for c in crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, REID_INPUT_SIZE)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors).to(self.device)
        batch = (batch - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(batch)
            feats = feats / torch.norm(feats, p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy()


class QualityFilter:
    def __init__(self, class_config: Dict[int, Dict[str, Any]] = CLASS_QUALITY_CONFIG):
        self.class_config = class_config

    def check_quality(self,
                      track: Track,
                      all_tracks: List[Track],
                      frame_shape: Tuple[int, int]) -> Tuple[QualityTier, str]:
        cfg = self.class_config.get(track.cls_id, self.class_config.get(0, {}))
        min_h = cfg.get("min_height", 80)
        min_w = cfg.get("min_width", 30)
        margin = cfg.get("border_margin", 5)
        max_iou = cfg.get("max_overlap_iou", 0.30)

        if track.height < min_h or track.width < min_w:
            return QualityTier.NONE, f"size {track.width}x{track.height} < {min_w}x{min_h}"

        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            if other.cls_id == track.cls_id:
                iou = compute_iou(track.box, other.box)
                if iou >= max_iou:
                    return QualityTier.NONE, f"overlap IoU {iou:.2f} >= {max_iou}"

        frame_h, frame_w = frame_shape[:2]
        touches_border = (track.x1 < margin or track.y1 < margin or
                          track.x2 > frame_w - margin or track.y2 > frame_h - margin)

        if touches_border:
            return QualityTier.PARTIAL, "touches border"

        return QualityTier.FULL, "ok"


# ==============================================================================
# DETECTOR & TRACKER WRAPPER
# ==============================================================================

class DetectorTracker:
    def __init__(self,
                 model_path: str = YOLO_MODEL_PATH,
                 device: Any = DEVICE,
                 classes: List[int] = TARGET_CLASSES,
                 conf_thresh: float = CONF_THRESHOLD,
                 tracker_config: str = TRACKER_CONFIG):
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_thresh = conf_thresh
        self.tracker_config = tracker_config

    def track(self, frame: np.ndarray) -> List[Track]:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.classes,
            conf=self.conf_thresh,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )

        tracks: List[Track] = []
        if not results or len(results) == 0:
            return tracks

        res = results[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return tracks

        box_coords = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.int().cpu().numpy()

        for box, track_id, conf, cls_id in zip(box_coords, track_ids, confs, classes):
            tracks.append(Track(track_id=track_id, box=box, conf=conf, cls_id=cls_id))

        return tracks


# ==============================================================================
# TARGET MANAGER
# ==============================================================================

class TargetManager:
    def __init__(self,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 quality_filter: Optional[QualityFilter] = None,
                 use_view_chain: bool = USE_VIEW_CHAIN_UPDATE,
                 use_color_cue: bool = USE_COLOR_CUE):
        self.state: TargetState = TargetState.IDLE
        self.feature_extractor = feature_extractor or PersonReIDExtractor()
        self.quality_filter = quality_filter or QualityFilter()

        self.use_view_chain = use_view_chain
        self.use_color_cue = use_color_cue

        # Galleries
        self.anchor_gallery: List[np.ndarray] = []
        self.anchor_color_hists: List[np.ndarray] = []
        self.dynamic_gallery: deque = deque(maxlen=DYNAMIC_GALLERY_MAX_SIZE)
        self.view_gallery: deque = deque(maxlen=VIEW_GALLERY_MAX_SIZE)  # E1

        self.last_gallery_add_frame: int = -999
        self.last_anchor_add_frame: int = -999
        self.last_view_chain_add_frame: int = -999
        self.last_suspicious_frame: int = -999
        self.last_person_near_frame: int = -999

        self.continuous_tracking_frames: int = 0
        self.target_recent_embs: deque = deque(maxlen=5)

        self.target_last_quality_emb: Optional[np.ndarray] = None
        self.target_track_id: Optional[int] = None
        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.last_center: Optional[Tuple[float, float]] = None
        self.velocity_history: deque = deque(maxlen=VELOCITY_SMOOTHING_WINDOW)
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_seen_frame: int = 0
        self.lost_frame: Optional[int] = None
        self.t_lost: float = 0.0

        self.missing_frames: int = 0
        self.fail_counter: int = 0
        self.frames_since_last_verify: int = 0
        self.suspicious_window_countdown: int = 0
        self.target_was_overlapping: bool = False

        self.candidate_history: Dict[int, Dict[str, Any]] = {}
        os.makedirs(GALLERY_DEBUG_DIR, exist_ok=True)

    def _transition_to(self, new_state: TargetState, frame_idx: int) -> None:
        if self.state == new_state:
            return
        old_state = self.state
        self.state = new_state
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{time_str} | Frame {frame_idx:05d}] State Change: {old_state.value} -> {new_state.value} "
              f"(Target ID: {self.target_track_id}, t_lost: {self.t_lost:.2f}s, "
              f"Anchors: {len(self.anchor_gallery)}, Dynamic: {len(self.dynamic_gallery)}, ViewChain: {len(self.view_gallery)})")

    def rollback_dynamic_gallery(self, current_frame: int, window: int = DYNAMIC_ROLLBACK_WINDOW) -> None:
        cutoff_frame = current_frame - window
        initial_dyn = len(self.dynamic_gallery)
        kept_dyn = [item for item in self.dynamic_gallery if item[2] < cutoff_frame]
        self.dynamic_gallery = deque(kept_dyn, maxlen=DYNAMIC_GALLERY_MAX_SIZE)

        initial_view = len(self.view_gallery)
        kept_view = [item for item in self.view_gallery if item[2] < cutoff_frame]
        self.view_gallery = deque(kept_view, maxlen=VIEW_GALLERY_MAX_SIZE)

        rem_dyn = initial_dyn - len(self.dynamic_gallery)
        rem_view = initial_view - len(self.view_gallery)
        if rem_dyn > 0 or rem_view > 0:
            print(f"[TargetManager] Rolled back galleries: removed {rem_dyn} dynamic and {rem_view} view-chain embeddings.")

    def compute_similarity_details(self,
                                   embedding: np.ndarray,
                                   color_hist: Optional[np.ndarray] = None) -> Dict[str, float]:
        def _calc_top3(emb_list, col_list):
            if not emb_list:
                return 0.0
            mat = np.array(emb_list)
            r_sims = np.dot(mat, embedding)
            if self.use_color_cue and color_hist is not None and col_list:
                c_mat = np.array(col_list)
                c_sims = np.dot(c_mat, color_hist)
                sims = REID_SIM_WEIGHT * r_sims + COLOR_CUE_WEIGHT * c_sims
            else:
                sims = r_sims
            k = min(TOP_K_GALLERY_SIM, len(sims))
            return float(np.mean(np.sort(sims)[-k:]))

        sim_anchor = _calc_top3(self.anchor_gallery, self.anchor_color_hists)

        dyn_embs = [item[0] for item in self.dynamic_gallery]
        dyn_cols = [item[1] for item in self.dynamic_gallery]
        sim_dyn = _calc_top3(dyn_embs, dyn_cols)

        view_embs = [item[0] for item in self.view_gallery]
        view_cols = [item[1] for item in self.view_gallery]

        all_embs = list(self.anchor_gallery) + dyn_embs + view_embs
        all_cols = list(self.anchor_color_hists) + dyn_cols + view_cols
        sim_all = _calc_top3(all_embs, all_cols)

        return {
            "sim_all": sim_all,
            "sim_anchor": sim_anchor,
            "sim_dynamic": sim_dyn
        }

    def select_target(self,
                      track_id: int,
                      box: Tuple[int, int, int, int],
                      frame_idx: int,
                      frame: Optional[np.ndarray] = None) -> None:
        self.target_track_id = track_id
        self.last_box = box
        self.last_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.velocity_history.clear()
        self.last_velocity = (0.0, 0.0)
        self.last_seen_frame = frame_idx
        self.lost_frame = None
        self.t_lost = 0.0
        self.missing_frames = 0
        self.fail_counter = 0
        self.frames_since_last_verify = 0
        self.suspicious_window_countdown = 0
        self.target_was_overlapping = False
        self.last_gallery_add_frame = -999
        self.last_anchor_add_frame = -999
        self.last_view_chain_add_frame = -999
        self.last_suspicious_frame = -999
        self.last_person_near_frame = -999
        self.continuous_tracking_frames = 0
        self.target_recent_embs.clear()
        self.target_last_quality_emb = None

        self.anchor_gallery.clear()
        self.anchor_color_hists.clear()
        self.dynamic_gallery.clear()
        self.view_gallery.clear()
        self.candidate_history.clear()

        if frame is not None:
            h, w = frame.shape[:2]
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(w, box[2]), min(h, box[3])
            margin = CLASS_QUALITY_CONFIG[0]["border_margin"]
            is_full = (x1 >= margin and y1 >= margin and x2 <= w - margin and y2 <= h - margin and
                       (x2 - x1) >= 30 and (y2 - y1) >= 80)
            if is_full:
                crop = frame[y1:y2, x1:x2]
                emb = self.feature_extractor.embed(crop)
                self.anchor_gallery.append(emb)
                if self.use_color_cue:
                    chist = compute_color_histogram(crop)
                    self.anchor_color_hists.append(chist)
                self.last_anchor_add_frame = frame_idx
                self.target_last_quality_emb = emb

                crop_path = os.path.join(GALLERY_DEBUG_DIR, f"anchor_{frame_idx:05d}.jpg")
                cv2.imwrite(crop_path, crop)

        if len(self.anchor_gallery) >= ANCHOR_GALLERY_SIZE:
            self._transition_to(TargetState.TRACKING, frame_idx)
        else:
            self._transition_to(TargetState.INITIALIZING, frame_idx)

    def update(self, tracks: List[Track], frame: np.ndarray, frame_idx: int, fps: float) -> Dict[str, Any]:
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        target_track: Optional[Track] = None
        target_score: Optional[float] = None
        target_sim_anchor: Optional[float] = None
        target_sim_dynamic: Optional[float] = None
        target_tier: QualityTier = QualityTier.NONE
        best_candidate_score: Optional[float] = None

        # LOGGING ONLY: diagnostics written to CSV, never read by any decision branch.
        log_fields: Dict[str, Any] = {
            "consec_sim": None,
            "fail_counter": None if self.state == TargetState.IDLE else self.fail_counter,
            "suspicious_countdown": None if self.state == TargetState.IDLE else self.suspicious_window_countdown,
            "should_verify": None,
            "passes_verify": None,
            "best_cand_id": None,
            "best_cand_tier": None,
            "best_cand_sim_all": None,
            "best_cand_sim_anchor": None,
            "best_cand_consec": None,
            "best_cand_req_accept": None,
            "best_cand_margin": None,
            "gate": None,
        }

        person_scores: Dict[int, float] = {}
        person_score_details: Dict[int, Dict[str, float]] = {}

        if self.state == TargetState.IDLE:
            log_fields["gate"] = "not_lost"
            return {
                "state": self.state.value,
                "target_box": None, "target_score": None, "sim_anchor": None, "sim_dynamic": None,
                "t_lost": 0.0, "frame_index": frame_idx, "best_candidate_score": None,
                "quality_tier": QualityTier.NONE.value, "person_scores": {}, "person_score_details": {},
                **log_fields
            }

        for t in tracks:
            if t.track_id == self.target_track_id:
                target_track = t
                break

        frame_h, frame_w = frame.shape[:2]
        track_tiers: Dict[int, QualityTier] = {}
        crops_dict: Dict[int, np.ndarray] = {}
        valid_crops: List[np.ndarray] = []
        valid_tids: List[int] = []

        for t in tracks:
            tier, _ = self.quality_filter.check_quality(t, tracks, (frame_h, frame_w))
            track_tiers[t.track_id] = tier

            x1, y1 = max(0, t.x1), max(0, t.y1)
            x2, y2 = min(frame_w, t.x2), min(frame_h, t.y2)
            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                crop = frame[y1:y2, x1:x2]
                crops_dict[t.track_id] = crop
                valid_crops.append(crop)
                valid_tids.append(t.track_id)

        embeddings: Dict[int, np.ndarray] = {}
        color_hists: Dict[int, np.ndarray] = {}
        if valid_crops:
            feats = self.feature_extractor.embed_batch(valid_crops)
            for tid, feat, crop in zip(valid_tids, feats, valid_crops):
                embeddings[tid] = feat
                if self.use_color_cue:
                    color_hists[tid] = compute_color_histogram(crop)

        if len(self.anchor_gallery) > 0:
            for tid, emb in embeddings.items():
                chist = color_hists.get(tid)
                s_dict = self.compute_similarity_details(emb, chist)
                person_scores[tid] = s_dict["sim_all"]
                person_score_details[tid] = s_dict

        # State: INITIALIZING
        if self.state == TargetState.INITIALIZING:
            if target_track is not None:
                self.last_box = target_track.box
                self.last_center = target_track.center
                self.last_seen_frame = frame_idx
                self.missing_frames = 0
                self.t_lost = 0.0
                target_tier = track_tiers.get(target_track.track_id, QualityTier.NONE)

                if target_tier in (QualityTier.FULL, QualityTier.PARTIAL) and target_track.track_id in embeddings:
                    emb = embeddings[target_track.track_id]
                    if self.target_last_quality_emb is not None:
                        consec_sim = float(np.dot(emb, self.target_last_quality_emb))
                        log_fields["consec_sim"] = consec_sim
                        if consec_sim < CONSECUTIVE_EMB_SIM_THRESH:
                            self.anchor_gallery.clear()
                            self.target_track_id = None
                            self.target_last_quality_emb = None
                            self._transition_to(TargetState.IDLE, frame_idx)
                            log_fields["gate"] = "not_lost"
                            return {
                                "state": self.state.value, "target_box": None, "target_score": None,
                                "sim_anchor": None, "sim_dynamic": None, "t_lost": 0.0,
                                "frame_index": frame_idx, "best_candidate_score": None,
                                "quality_tier": QualityTier.NONE.value,
                                "person_scores": person_scores, "person_score_details": person_score_details,
                                **log_fields
                            }
                    self.target_last_quality_emb = emb

                if (target_tier == QualityTier.FULL and target_track.track_id in embeddings and
                        frame_idx > self.last_anchor_add_frame and
                        (frame_idx - self.last_anchor_add_frame) >= ANCHOR_FRAME_SPACING):

                    emb = embeddings[target_track.track_id]
                    max_sim_anchor = max([float(np.dot(a, emb)) for a in self.anchor_gallery]) if self.anchor_gallery else 0.0

                    if max_sim_anchor <= ANCHOR_MAX_SIM:
                        self.anchor_gallery.append(emb)
                        if self.use_color_cue:
                            self.anchor_color_hists.append(color_hists[target_track.track_id])
                        self.last_anchor_add_frame = frame_idx

                        crop_path = os.path.join(GALLERY_DEBUG_DIR, f"anchor_{frame_idx:05d}.jpg")
                        cv2.imwrite(crop_path, crops_dict[target_track.track_id])

                        if len(self.anchor_gallery) >= ANCHOR_GALLERY_SIZE:
                            self._transition_to(TargetState.TRACKING, frame_idx)

                tgt_s = person_score_details.get(target_track.track_id, {})
                target_score = tgt_s.get("sim_all", 1.0)
                target_sim_anchor = tgt_s.get("sim_anchor", 1.0)
                target_sim_dynamic = tgt_s.get("sim_dynamic", 0.0)

            else:
                self.missing_frames += 1
                if self.missing_frames >= MISSING_FRAMES_FOR_LOST:
                    if len(self.anchor_gallery) >= 3:
                        self.lost_frame = self.last_seen_frame
                        self.t_lost = (frame_idx - self.lost_frame) / fps if fps > 0 else 0.0
                        self._transition_to(TargetState.LOST, frame_idx)
                    else:
                        self._transition_to(TargetState.IDLE, frame_idx)

        # State: TRACKING
        elif self.state == TargetState.TRACKING:
            if target_track is not None:
                self.continuous_tracking_frames += 1
                curr_center = target_track.center
                curr_box = target_track.box
                curr_area = target_track.area
                target_tier = track_tiers.get(target_track.track_id, QualityTier.NONE)

                target_is_overlapping = any(
                    compute_iou(curr_box, o.box) > OVERLAP_IOU_THRESHOLD
                    for o in tracks if o.track_id != target_track.track_id
                )

                has_near_person = any(
                    compute_iou(curr_box, o.box) > 0.10
                    for o in tracks if o.track_id != target_track.track_id
                )
                if has_near_person:
                    self.last_person_near_frame = frame_idx

                if target_tier in (QualityTier.FULL, QualityTier.PARTIAL) and target_track.track_id in embeddings:
                    emb = embeddings[target_track.track_id]
                    self.target_recent_embs.append(emb)

                    if self.target_last_quality_emb is not None:
                        consec_sim = float(np.dot(emb, self.target_last_quality_emb))
                        log_fields["consec_sim"] = consec_sim
                        if consec_sim < CONSECUTIVE_EMB_SIM_THRESH:
                            log_fields["fail_counter"] = self.fail_counter
                            log_fields["suspicious_countdown"] = self.suspicious_window_countdown
                            self.rollback_dynamic_gallery(frame_idx, DYNAMIC_ROLLBACK_WINDOW)
                            self.lost_frame = frame_idx
                            self.t_lost = 0.0
                            self.fail_counter = 0
                            self.suspicious_window_countdown = 0
                            self.target_last_quality_emb = None
                            self.continuous_tracking_frames = 0
                            self._transition_to(TargetState.LOST, frame_idx)
                            return {
                                "state": self.state.value, "target_box": None, "target_score": None,
                                "sim_anchor": None, "sim_dynamic": None, "t_lost": 0.0,
                                "frame_index": frame_idx, "best_candidate_score": None,
                                "quality_tier": target_tier.value,
                                "person_scores": person_scores, "person_score_details": person_score_details,
                                **log_fields
                            }
                    self.target_last_quality_emb = emb

                trigger_suspicious = False
                suspicious_reasons = []

                if self.last_center is not None:
                    center_jump = np.hypot(curr_center[0] - self.last_center[0], curr_center[1] - self.last_center[1])
                    if center_jump > (CENTER_JUMP_RATIO_THRESH * frame_w):
                        trigger_suspicious = True
                        suspicious_reasons.append(f"center jump {center_jump:.1f}px")

                if self.last_box is not None:
                    prev_w = max(1, self.last_box[2] - self.last_box[0])
                    prev_h = max(1, self.last_box[3] - self.last_box[1])
                    prev_area = prev_w * prev_h
                    ratio = curr_area / prev_area
                    if ratio > AREA_CHANGE_RATIO_THRESH or ratio < (1.0 / AREA_CHANGE_RATIO_THRESH):
                        trigger_suspicious = True
                        suspicious_reasons.append(f"area ratio {ratio:.2f}x")

                if self.target_was_overlapping and not target_is_overlapping:
                    trigger_suspicious = True
                    suspicious_reasons.append("overlap ended")

                tgt_details = person_score_details.get(target_track.track_id, {})
                target_score = tgt_details.get("sim_all")
                target_sim_anchor = tgt_details.get("sim_anchor")
                target_sim_dynamic = tgt_details.get("sim_dynamic")

                if target_sim_anchor is not None and target_sim_anchor < T_ANCHOR_FLOOR:
                    trigger_suspicious = True
                    suspicious_reasons.append(f"anchor sim {target_sim_anchor:.2f} < {T_ANCHOR_FLOOR}")

                self.target_was_overlapping = target_is_overlapping

                if trigger_suspicious:
                    self.suspicious_window_countdown = max(self.suspicious_window_countdown, SUSPICIOUS_WINDOW_FRAMES)
                    self.last_suspicious_frame = frame_idx

                is_suspicious_mode = (self.suspicious_window_countdown > 0)
                is_tier_ok_for_verify = (target_tier in (QualityTier.FULL, QualityTier.PARTIAL))
                should_verify = False

                if is_suspicious_mode:
                    if is_tier_ok_for_verify:
                        should_verify = True
                    self.suspicious_window_countdown -= 1
                else:
                    self.frames_since_last_verify += 1
                    if self.frames_since_last_verify >= VERIFY_INTERVAL_FRAMES and is_tier_ok_for_verify:
                        should_verify = True

                log_fields["should_verify"] = should_verify
                log_fields["suspicious_countdown"] = self.suspicious_window_countdown

                if should_verify:
                    self.frames_since_last_verify = 0
                    if target_score is not None and target_sim_anchor is not None:
                        passes_verify = (target_score >= T_VERIFY and target_sim_anchor >= T_ANCHOR_FLOOR)
                        log_fields["passes_verify"] = passes_verify
                        if passes_verify:
                            self.fail_counter = 0
                            log_fields["fail_counter"] = self.fail_counter
                        else:
                            self.fail_counter += 1
                            log_fields["fail_counter"] = self.fail_counter
                            fail_limit = SUSPICIOUS_FAIL_LIMIT if is_suspicious_mode else VERIFY_FAIL_LIMIT
                            if self.fail_counter >= fail_limit:
                                self.rollback_dynamic_gallery(frame_idx, DYNAMIC_ROLLBACK_WINDOW)
                                self.lost_frame = frame_idx
                                self.t_lost = 0.0
                                self.fail_counter = 0
                                self.suspicious_window_countdown = 0
                                self.target_last_quality_emb = None
                                self.continuous_tracking_frames = 0
                                self._transition_to(TargetState.LOST, frame_idx)

                # Standard Dynamic gallery update
                if self.state == TargetState.TRACKING and target_tier == QualityTier.FULL and not target_is_overlapping:
                    if target_score is not None and target_sim_anchor is not None:
                        can_add_dynamic = (
                            target_sim_anchor >= DYNAMIC_ADD_ANCHOR_MIN and
                            (frame_idx - self.last_suspicious_frame >= DYNAMIC_ADD_SUSPICIOUS_SPACING) and
                            target_score >= T_UPDATE and
                            (frame_idx - self.last_gallery_add_frame >= MIN_FRAMES_BETWEEN_ADDITIONS)
                        )
                        if can_add_dynamic:
                            emb = embeddings[target_track.track_id]
                            all_embs = list(self.anchor_gallery) + [item[0] for item in self.dynamic_gallery]
                            max_sim = float(np.max(np.dot(np.array(all_embs), emb))) if all_embs else 0.0
                            if max_sim < MAX_SIM_FOR_ADDITION:
                                col_h = color_hists.get(target_track.track_id)
                                self.dynamic_gallery.append((emb, col_h, frame_idx))
                                self.last_gallery_add_frame = frame_idx

                                crop_path = os.path.join(GALLERY_DEBUG_DIR, f"dyn_{frame_idx:05d}.jpg")
                                cv2.imwrite(crop_path, crops_dict[target_track.track_id])

                # EXPERIMENT E1: USE_VIEW_CHAIN_UPDATE
                if (self.use_view_chain and self.state == TargetState.TRACKING and
                        target_tier in (QualityTier.FULL, QualityTier.PARTIAL)):
                    if target_sim_anchor is not None and T_ANCHOR_FLOOR <= target_sim_anchor < 0.80:
                        can_chain_add = (
                            self.continuous_tracking_frames >= 60 and
                            (frame_idx - self.last_suspicious_frame >= 30) and
                            (frame_idx - self.last_person_near_frame >= 30) and
                            (frame_idx - self.last_view_chain_add_frame >= 15) and
                            len(self.target_recent_embs) >= 5
                        )
                        if can_chain_add:
                            emb = embeddings[target_track.track_id]
                            recent_mat = np.array(list(self.target_recent_embs))
                            mean_recent = np.mean(recent_mat, axis=0)
                            mean_recent = mean_recent / (np.linalg.norm(mean_recent) + 1e-6)
                            chain_sim = float(np.dot(emb, mean_recent))

                            if chain_sim >= 0.80:
                                col_h = color_hists.get(target_track.track_id)
                                self.view_gallery.append((emb, col_h, frame_idx))
                                self.last_view_chain_add_frame = frame_idx

                                crop_path = os.path.join(GALLERY_DEBUG_DIR, f"view_chain_{frame_idx:05d}.jpg")
                                cv2.imwrite(crop_path, crops_dict[target_track.track_id])
                                print(f"[{time_str} | Frame {frame_idx:05d}] View Chain Gallery Addition (saved {crop_path}) | "
                                      f"View gallery size: {len(self.view_gallery)}/{VIEW_GALLERY_MAX_SIZE}")

                if self.state == TargetState.TRACKING:
                    if self.last_center is not None:
                        vx = curr_center[0] - self.last_center[0]
                        vy = curr_center[1] - self.last_center[1]
                        self.velocity_history.append((vx, vy))
                        avg_vx = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
                        avg_vy = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)
                        self.last_velocity = (avg_vx, avg_vy)

                    self.last_box = curr_box
                    self.last_center = curr_center
                    self.last_seen_frame = frame_idx
                    self.missing_frames = 0
                    self.t_lost = 0.0

            else:
                self.missing_frames += 1
                if self.missing_frames >= MISSING_FRAMES_FOR_LOST:
                    self.lost_frame = self.last_seen_frame
                    self.t_lost = (frame_idx - self.lost_frame) / fps if fps > 0 else 0.0
                    self.fail_counter = 0
                    self.suspicious_window_countdown = 0
                    self.target_last_quality_emb = None
                    self.continuous_tracking_frames = 0
                    self._transition_to(TargetState.LOST, frame_idx)

        # State: LOST
        elif self.state == TargetState.LOST:
            self.continuous_tracking_frames = 0
            if self.lost_frame is not None:
                self.t_lost = (frame_idx - self.lost_frame) / fps if fps > 0 else 0.0

            if self.t_lost > T_MAX_LOST:
                self._transition_to(TargetState.GIVE_UP, frame_idx)
            else:
                for t in tracks:
                    tid = t.track_id
                    tier = track_tiers.get(tid, QualityTier.NONE)

                    if tier in (QualityTier.FULL, QualityTier.PARTIAL) and tid in embeddings:
                        emb = embeddings[tid]
                        t_data = self.candidate_history.setdefault(
                            tid, {"consecutive_ok_count": 0, "scores": [], "last_quality_emb": None}
                        )

                        if t_data["last_quality_emb"] is not None:
                            consec_sim = float(np.dot(emb, t_data["last_quality_emb"]))
                            if consec_sim < CONSECUTIVE_EMB_SIM_THRESH:
                                t_data["consecutive_ok_count"] = 0
                                t_data["scores"].clear()
                        t_data["last_quality_emb"] = emb

                        s_dict = person_score_details.get(tid, {})
                        s_all = s_dict.get("sim_all", 0.0)
                        s_anchor = s_dict.get("sim_anchor", 0.0)

                        base_accept = T_ACCEPT_LONG
                        req_accept = base_accept + (PARTIAL_ACCEPT_PENALTY if tier == QualityTier.PARTIAL else 0.0)

                        passes_req = (s_all >= req_accept and s_anchor >= T_ANCHOR_FLOOR)
                        if passes_req:
                            t_data["consecutive_ok_count"] += 1
                            t_data["scores"].append(s_all)
                        else:
                            t_data["consecutive_ok_count"] = 0
                            t_data["scores"].clear()

                qualifying_candidates = []
                current_tids = {t.track_id for t in tracks}

                for t in tracks:
                    tid = t.track_id
                    t_data = self.candidate_history.get(tid)
                    if t_data and t_data["consecutive_ok_count"] >= N_CONFIRM:
                        c_mean = float(np.mean(t_data["scores"][-N_CONFIRM:]))

                        other_scores = [
                            person_scores[oid] for oid in current_tids
                            if oid != tid and oid in person_scores
                        ]
                        if other_scores:
                            best_other = max(other_scores)
                            if (c_mean - best_other) < MARGIN_MIN:
                                continue

                        qualifying_candidates.append((t, c_mean))

                if qualifying_candidates:
                    best_cand, best_mean = max(qualifying_candidates, key=lambda item: item[1])
                    self.target_track_id = best_cand.track_id
                    self.last_box = best_cand.box
                    self.last_center = best_cand.center
                    self.last_seen_frame = frame_idx
                    self.missing_frames = 0
                    self.fail_counter = 0
                    self.suspicious_window_countdown = SUSPICIOUS_WINDOW_FRAMES
                    self.last_suspicious_frame = frame_idx
                    self.target_was_overlapping = False
                    self.target_last_quality_emb = embeddings.get(best_cand.track_id)
                    self.continuous_tracking_frames = 0
                    self._transition_to(TargetState.TRACKING, frame_idx)
                    self.t_lost = 0.0
                    self.candidate_history.clear()

                # LOGGING ONLY: per-frame diagnostics of the best non-target candidate.
                # These values are written to CSV and never read by any decision branch.
                if self.state == TargetState.LOST:
                    non_target_tids = [t.track_id for t in tracks if t.track_id != self.target_track_id]
                    scored_tids = [tid for tid in non_target_tids if tid in person_scores]
                    if not non_target_tids:
                        log_fields["gate"] = "no_candidate"
                    elif not scored_tids:
                        log_fields["gate"] = "tier_none"
                    else:
                        b_tid = max(scored_tids, key=lambda tid: person_scores[tid])
                        b_tier = track_tiers.get(b_tid, QualityTier.NONE)
                        b_det = person_score_details.get(b_tid, {})
                        b_all = b_det.get("sim_all")
                        b_anchor = b_det.get("sim_anchor")
                        b_data = self.candidate_history.get(b_tid)
                        b_consec = b_data["consecutive_ok_count"] if b_data is not None else 0
                        b_req = T_ACCEPT_LONG + (PARTIAL_ACCEPT_PENALTY if b_tier == QualityTier.PARTIAL else 0.0)
                        other_scores = [
                            person_scores[oid] for oid in non_target_tids
                            if oid != b_tid and oid in person_scores
                        ]
                        b_margin = None
                        if (b_data is not None and b_consec >= N_CONFIRM and b_data["scores"] and other_scores):
                            b_margin = float(np.mean(b_data["scores"][-N_CONFIRM:])) - max(other_scores)

                        log_fields["best_cand_id"] = b_tid
                        log_fields["best_cand_tier"] = b_tier.value
                        log_fields["best_cand_sim_all"] = b_all
                        log_fields["best_cand_sim_anchor"] = b_anchor
                        log_fields["best_cand_consec"] = b_consec
                        log_fields["best_cand_req_accept"] = b_req
                        log_fields["best_cand_margin"] = b_margin

                        if b_tier not in (QualityTier.FULL, QualityTier.PARTIAL) or b_tid not in embeddings:
                            log_fields["gate"] = "tier_none"
                        elif b_all is None or b_all < b_req:
                            log_fields["gate"] = "accept_fail"
                        elif b_anchor is None or b_anchor < T_ANCHOR_FLOOR:
                            log_fields["gate"] = "floor_fail"
                        elif b_consec < N_CONFIRM:
                            log_fields["gate"] = "nconfirm_fail"
                        elif b_margin is not None and b_margin < MARGIN_MIN:
                            log_fields["gate"] = "margin_fail"
                        else:
                            log_fields["gate"] = ""

        # State: GIVE_UP
        elif self.state == TargetState.GIVE_UP:
            self.continuous_tracking_frames = 0
            if self.lost_frame is not None:
                self.t_lost = (frame_idx - self.lost_frame) / fps if fps > 0 else 0.0

        if self.state != TargetState.LOST and log_fields["gate"] is None:
            log_fields["gate"] = "not_lost"

        non_target_scores = [s for tid, s in person_scores.items() if tid != self.target_track_id]
        if non_target_scores:
            best_candidate_score = max(non_target_scores)

        output_box = None
        if self.state in (TargetState.TRACKING, TargetState.INITIALIZING) and target_track is not None:
            if self.state == TargetState.INITIALIZING or (target_sim_anchor is not None and target_sim_anchor >= T_ANCHOR_FLOOR):
                output_box = target_track.box

        return {
            "state": self.state.value,
            "target_box": output_box,
            "target_score": target_score,
            "sim_anchor": target_sim_anchor,
            "sim_dynamic": target_sim_dynamic,
            "t_lost": self.t_lost,
            "frame_index": frame_idx,
            "best_candidate_score": best_candidate_score,
            "quality_tier": target_tier.value,
            "person_scores": person_scores,
            "person_score_details": person_score_details,
            **log_fields
        }


# ==============================================================================
# VISUALIZATION & HUD
# ==============================================================================

def draw_hud(frame: np.ndarray,
             manager_output: Dict[str, Any],
             tracks: List[Track],
             target_track_id: Optional[int],
             fps: float,
             suspicious_frames_left: int = 0) -> np.ndarray:
    vis_frame = frame.copy()
    state = manager_output["state"]
    target_box = manager_output["target_box"]
    t_lost = manager_output["t_lost"]
    person_scores = manager_output.get("person_scores", {})
    sim_anchor = manager_output.get("sim_anchor")
    target_score = manager_output.get("target_score")

    for t in tracks:
        is_target_box = (t.track_id == target_track_id and
                         state in (TargetState.TRACKING.value, TargetState.INITIALIZING.value) and
                         target_box is not None)
        if is_target_box:
            continue

        sim = person_scores.get(t.track_id)
        sim_str = f" | {sim:.2f}" if sim is not None else ""
        label = f"ID: {t.track_id}{sim_str}"

        cv2.rectangle(vis_frame, (t.x1, t.y1), (t.x2, t.y2), (128, 128, 128), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_frame, (t.x1, max(0, t.y1 - th - 6)), (t.x1 + tw + 6, t.y1), (128, 128, 128), -1)
        cv2.putText(vis_frame, label, (t.x1 + 3, t.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if state in (TargetState.TRACKING.value, TargetState.INITIALIZING.value) and target_box is not None:
        x1, y1, x2, y2 = target_box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if state == TargetState.INITIALIZING.value:
            anc_str = f" | anchor {sim_anchor:.2f}" if sim_anchor is not None else ""
            label = f"TARGET [ID: {target_track_id}] (INIT){anc_str}"
        elif target_score is not None and sim_anchor is not None:
            label = f"TARGET [ID: {target_track_id}] | all {target_score:.2f} | anchor {sim_anchor:.2f}"
        else:
            label = f"TARGET [ID: {target_track_id}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(vis_frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, vis_frame, 0.28, 0, vis_frame)
    cv2.rectangle(vis_frame, (10, 10), (370, 140), (70, 70, 70), 1)

    state_color = (0, 255, 0) if state == "TRACKING" else (0, 255, 255) if state in ("LOST", "INITIALIZING") else (0, 0, 255)
    cv2.putText(vis_frame, f"STATE: {state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
    cv2.putText(vis_frame, f"TARGET ID: {target_track_id if target_track_id is not None else 'None'}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(vis_frame, f"FPS: {fps:5.1f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if state in (TargetState.LOST.value, TargetState.GIVE_UP.value):
        cv2.putText(vis_frame, f"TIME LOST: {t_lost:.2f} s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    elif suspicious_frames_left > 0:
        cv2.putText(vis_frame, f"VERIFY: HIGH-FREQ ({suspicious_frames_left}f)", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    else:
        cv2.putText(vis_frame, f"PEOPLE VISIBLE: {len(tracks)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return vis_frame


# ==============================================================================
# INTERACTIVE TARGET SELECTION
# ==============================================================================

def select_target_interactive(cap: cv2.VideoCapture,
                               detector: DetectorTracker,
                               window_name: str = "Select Target") -> Tuple[int, Tuple[int, int, int, int], int, np.ndarray, List[Track]]:
    clicked_pt = [None]
    confirmed = [False]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pt[0] = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    selected_track: Optional[Track] = None
    frame_idx = 0
    current_frame: Optional[np.ndarray] = None
    tracks: List[Track] = []

    while True:
        if current_frame is None:
            ret, frame = cap.read()
            if not ret:
                cv2.destroyWindow(window_name)
                raise RuntimeError("Video ended before target was selected.")
            current_frame = frame
            tracks = detector.track(current_frame)

        display = current_frame.copy()

        if clicked_pt[0] is not None:
            candidates = [t for t in tracks if t.contains_point(clicked_pt[0])]
            if candidates:
                selected_track = min(candidates, key=lambda t: t.area)
            clicked_pt[0] = None

        for t in tracks:
            is_selected = selected_track is not None and t.track_id == selected_track.track_id
            color = (0, 255, 0) if is_selected else (128, 128, 128)
            thickness = 3 if is_selected else 2
            cv2.rectangle(display, (t.x1, t.y1), (t.x2, t.y2), color, thickness)
            label = f"TARGET [ID: {t.track_id}]" if is_selected else f"ID: {t.track_id}"
            cv2.putText(display, label, (t.x1, max(20, t.y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        banner = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        if selected_track is not None:
            msg = f"Track ID {selected_track.track_id} selected. Press SPACE/ENTER to start."
            text_color = (0, 255, 0)
        elif len(tracks) > 0:
            msg = f"Frame {frame_idx}: Click on a person to select target. ('n' for next, 'q' to quit)"
            text_color = (255, 255, 255)
        else:
            msg = f"Frame {frame_idx}: No people detected. Press 'n' to advance frame, 'q' to quit."
            text_color = (0, 165, 255)

        cv2.putText(banner, msg, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        combined = np.vstack([banner, display])
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF
        if key in (32, 13):
            if selected_track is not None:
                confirmed[0] = True
                break
        elif key == ord('n'):
            current_frame = None
            frame_idx += 1
        elif key in (ord('q'), 27):
            cv2.destroyWindow(window_name)
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return selected_track.track_id, selected_track.box, frame_idx, current_frame, tracks


# ==============================================================================
# MAIN PROCESSING PIPELINE
# ==============================================================================

def run_tracking(video_path: str,
                 out_path: str,
                 target_id: Optional[int] = None,
                 headless: bool = False,
                 use_view_chain: bool = False,
                 use_color_cue: bool = False,
                 reid_model: str = REID_MODEL_NAME) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    logs_dir = "/home/foxsec/drone_project/logs"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(GALLERY_DEBUG_DIR, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv_path = os.path.join(logs_dir, f"tracking_log_{timestamp_str}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    detector = DetectorTracker()

    # Model weights mapping
    REID_WEIGHTS_MAP = {
        "osnet_x1_0":     "/home/foxsec/drone_project/weights/osnet_x1_0_msmt17.pth",
        "osnet_ain_x1_0": "/home/foxsec/drone_project/weights/osnet_ain_x1_0_msmt17.pth",
    }
    weights_path = REID_WEIGHTS_MAP.get(reid_model, REID_WEIGHTS_PATH)

    reid_extractor = PersonReIDExtractor(model_name=reid_model, weights_path=weights_path)
    manager = TargetManager(
        feature_extractor=reid_extractor,
        use_view_chain=use_view_chain,
        use_color_cue=use_color_cue
    )

    first_frame: Optional[np.ndarray] = None
    first_tracks: List[Track] = []

    if target_id is not None:
        start_frame_idx = 0
        manager.target_track_id = target_id
    else:
        if headless:
            raise ValueError("Interactive click selection requires a display. Provide --target-id in headless mode.")
        selected_id, selected_box, start_frame_idx, first_frame, first_tracks = select_target_interactive(cap, detector)
        manager.select_target(selected_id, selected_box, start_frame_idx, first_frame)

    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("frame,state,target_track_id,target_score,sim_anchor,sim_dynamic,best_candidate_score,quality_tier,num_people,"
                "consec_sim,fail_counter,suspicious_countdown,should_verify,passes_verify,best_cand_id,best_cand_tier,"
                "best_cand_sim_all,best_cand_sim_anchor,best_cand_consec,best_cand_req_accept,best_cand_margin,gate\n")

    window_name = f"Drone Tracking ({reid_model})"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = start_frame_idx
    processed_count = 0
    start_time = time.time()
    current_fps = fps
    suspected_false_lock_frames: List[int] = []

    try:
        while True:
            t0 = time.time()

            if first_frame is not None:
                frame = first_frame
                tracks = first_tracks if first_tracks else detector.track(frame)
                first_frame = None
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                tracks = detector.track(frame)

            if manager.state == TargetState.IDLE and manager.target_track_id is not None:
                for t in tracks:
                    if t.track_id == manager.target_track_id:
                        manager.select_target(t.track_id, t.box, frame_count, frame)
                        break

            output = manager.update(tracks, frame, frame_count, fps)

            if output["state"] == TargetState.TRACKING.value:
                s_anc = output["sim_anchor"]
                if s_anc is not None and s_anc < T_ANCHOR_FLOOR:
                    suspected_false_lock_frames.append(frame_count)

            with open(log_csv_path, "a", encoding="utf-8") as f:
                tgt_score_str = f"{output['target_score']:.4f}" if output['target_score'] is not None else ""
                sim_anc_str = f"{output['sim_anchor']:.4f}" if output['sim_anchor'] is not None else ""
                sim_dyn_str = f"{output['sim_dynamic']:.4f}" if output['sim_dynamic'] is not None else ""
                cand_score_str = f"{output['best_candidate_score']:.4f}" if output['best_candidate_score'] is not None else ""
                tgt_id_str = str(manager.target_track_id) if manager.target_track_id is not None else ""
                tier_str = output['quality_tier']

                def _log_str(val) -> str:
                    if val is None:
                        return ""
                    if isinstance(val, bool):
                        return str(int(val))
                    if isinstance(val, float):
                        return f"{val:.4f}"
                    return str(val)

                extra_cols = [_log_str(output.get(k)) for k in (
                    "consec_sim", "fail_counter", "suspicious_countdown", "should_verify", "passes_verify",
                    "best_cand_id", "best_cand_tier", "best_cand_sim_all", "best_cand_sim_anchor",
                    "best_cand_consec", "best_cand_req_accept", "best_cand_margin", "gate"
                )]
                f.write(f"{frame_count},{output['state']},{tgt_id_str},{tgt_score_str},"
                        f"{sim_anc_str},{sim_dyn_str},{cand_score_str},{tier_str},{len(tracks)},"
                        f"{','.join(extra_cols)}\n")

            elapsed = time.time() - t0
            inst_fps = 1.0 / elapsed if elapsed > 0 else fps
            current_fps = 0.9 * current_fps + 0.1 * inst_fps

            annotated_frame = draw_hud(
                frame, output, tracks, manager.target_track_id, current_fps,
                suspicious_frames_left=manager.suspicious_window_countdown
            )
            writer.write(annotated_frame)

            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

            frame_count += 1
            processed_count += 1

    finally:
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0.0
        cap.release()
        writer.release()
        if not headless:
            cv2.destroyAllWindows()

        # Format suspected false lock frame ranges (R6)
        def _group_ranges(frames):
            if not frames:
                return []
            ranges = []
            start = frames[0]
            end = frames[0]
            for f in frames[1:]:
                if f == end + 1:
                    end = f
                else:
                    ranges.append(f"{start}-{end}" if start != end else f"{start}")
                    start = f
                    end = f
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            return ranges

        false_lock_ranges = _group_ranges(suspected_false_lock_frames)

        print("\n" + "=" * 60)
        print("TRACKING RUN COMPLETED (STEP 2 REFINED)")
        print(f"  ReID Weights File       : {os.path.basename(weights_path)}")
        print(f"  Start frame index       : {start_frame_idx}")
        print(f"  Frames processed        : {processed_count}")
        print(f"  Total processing time   : {total_time:.2f} s")
        print(f"  Average pipeline FPS    : {avg_fps:.2f} FPS")
        print(f"  Annotated video saved   : {out_path}")
        print(f"  CSV log saved           : {log_csv_path}")
        print(f"  Gallery debug images    : {GALLERY_DEBUG_DIR}")
        print(f"  Suspected False-Locks   : {false_lock_ranges if false_lock_ranges else 'None (0 frames)'}")
        print("=" * 60 + "\n")


# ==============================================================================


# ==============================================================================
# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous Drone Single-Target Tracking (Step 2/Step 3).")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out", type=str, default="/home/foxsec/drone_project/outputs/output.mp4", help="Path to output video")
    parser.add_argument("--target-id", type=int, default=None, help="Pre-select target track ID")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--use-view-chain", action="store_true", help="Enable E1: View chain updates")
    parser.add_argument("--use-color-cue", action="store_true", help="Enable E2: HSV color histogram cue")
    parser.add_argument("--reid-model", type=str, default=REID_MODEL_NAME, help="E3: ReID model name (osnet_x1_0, osnet_ain_x1_0)")

    args = parser.parse_args()
    run_tracking(
        video_path=args.video,
        out_path=args.out,
        target_id=args.target_id,
        headless=args.headless,
        use_view_chain=args.use_view_chain,
        use_color_cue=args.use_color_cue,
        reid_model=args.reid_model
    )


if __name__ == "__main__":
    main()
