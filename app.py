from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

# Libraries for model inference and MediaPipe
import torch
import torch.nn as nn
import numpy as np
import os
import pathlib, platform
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PurePosixPath
import random
import json
import datetime
import threading
import re
import cv2
import mediapipe as mp

# If converted to TensorRT, therefore having model.engine in the folder.
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit          # initialises CUDA context
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

from model import ModifiedLSTM
from pathutils import resource_path

app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static"),
)

CORS(app)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    os.getenv("DATABASE_URL") 
    or "sqlite:///fsl.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    results = db.relationship("PracticeResult", backref="user", lazy=True)

class PracticeResult(db.Model):
    __tablename__ = "practice_results"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    label = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

def init_db():
    with app.app_context():
        db.create_all()

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def save_progress(label: str, confidence=None):
    """Save a practice result to PostgreSQL if the user is logged in."""
    uid = session.get("user_id")
    if not uid:
        return
    row = PracticeResult(
        user_id=uid,
        label=label,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
    )
    db.session.add(row)
    db.session.commit()

# For model loading and inference
MODEL_PATH = r"run20.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

checkpoint = torch.load( 
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]

CLASSES = config["CLASSES"]
INPUT_SIZE = config["FEATURE_DIM"]
HIDDEN_SIZE = config["HIDDEN_SIZE"]
NUM_LAYERS = config["NUM_LAYERS"]
DROPOUT = config["DROPOUT"]
SEQ_LEN = config.get("SEQ_LEN", config.get("SEQUENCE_LENGTH", 48))

model = ModifiedLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    len(CLASSES),
    dropout=DROPOUT,
    use_layernorm=True
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

if device.type == "cuda":
    model.half()

model.eval()

# Get tensor dtypes for TensorRT engine if available
TRT_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "model.engine")
use_trt = False

# Helper: map TensorRT dtype enum → numpy dtype
def _trt_dtype_to_np(trt_dtype):
    """Convert a TensorRT DataType to the corresponding numpy dtype."""
    _map = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8:    np.int8,
        trt.int32:   np.int32,
    }
    return _map.get(trt_dtype, np.float32)

if HAS_TRT and os.path.exists(TRT_ENGINE_PATH):
    print(f"Loading TensorRT engine from {TRT_ENGINE_PATH} ...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(TRT_ENGINE_PATH, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    trt_context = engine.create_execution_context()

    trt_input_name  = engine.get_tensor_name(0)
    trt_output_name = engine.get_tensor_name(1)

    # Detects dtypes of memory.engine if its fp16 or fp32
    trt_input_dtype  = _trt_dtype_to_np(engine.get_tensor_dtype(trt_input_name))
    trt_output_dtype = _trt_dtype_to_np(engine.get_tensor_dtype(trt_output_name))

    input_shape  = (1, SEQ_LEN, INPUT_SIZE)
    output_shape = (1, len(CLASSES))

    input_nbytes  = int(np.prod(input_shape)  * np.dtype(trt_input_dtype).itemsize)
    output_nbytes = int(np.prod(output_shape) * np.dtype(trt_output_dtype).itemsize)

    d_input  = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)
    trt_stream = cuda.Stream()

    use_trt = True
    print(f"TensorRT engine loaded — input dtype={trt_input_dtype}, output dtype={trt_output_dtype}")
else:
    trt_input_dtype  = np.float32
    trt_output_dtype = np.float32
    if not HAS_TRT:
        print("TensorRT / PyCUDA not installed — using PyTorch.")
    else:
        print("TensorRT engine not found. Using PyTorch.")

print("Loaded Config:")
print("Hidden:", HIDDEN_SIZE)
print("Layers:", NUM_LAYERS)
print("Dropout:", DROPOUT)
print("Classes:", len(CLASSES))
print("SEQ_LEN from config:", SEQ_LEN)
print(f"[APP] Loaded model → hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT}")

_mp_holistic_mod = mp.solutions.holistic

server_holistic = _mp_holistic_mod.Holistic(
    static_image_mode=False,
    model_complexity= 2,          
    smooth_landmarks=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

# Separate Holistic for ghost/demo video processing — static mode so each
# frame is detected independently (no cross-contamination with live tracker)
_ghost_holistic = _mp_holistic_mod.Holistic(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
_ghost_lock = threading.Lock()  # protect ghost holistic separately

# Face detector for counting people in frame
_mp_face_det_mod = mp.solutions.face_detection
server_face_detector = _mp_face_det_mod.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5,
)

_mp_lock = threading.Lock() 
_face_count_cache = 0        # cached person count
_face_frame_counter = 0      # runs face detection every N frames

def _serialize_landmarks(landmark_list):
    """Convert a MediaPipe NormalizedLandmarkList → list of dicts."""
    if landmark_list is None:
        return None
    return [
        {"x": l.x, "y": l.y, "z": l.z,
         "visibility": getattr(l, "visibility", 0.0)}
        for l in landmark_list.landmark
    ]

print("[APP] Server-side MediaPipe Holistic initialised")

def prepare_sequence(data_json):
    seq_len = SEQ_LEN
    feat_dim = INPUT_SIZE

    if "sequence" in data_json:
        seq = np.array(data_json["sequence"], dtype=np.float32)
        print(f"[DEBUG] sequence ndim={seq.ndim}, shape={seq.shape}, size={seq.size}, expected 1D size={seq_len*feat_dim}")

        if seq.ndim == 1 and seq.size == seq_len * feat_dim:
            seq = seq.reshape(seq_len, feat_dim)

        elif seq.ndim == 2:
            if seq.shape != (seq_len, feat_dim):
                raise ValueError(f"sequence shape {seq.shape}, expected {(seq_len, feat_dim)}")

        else:
            raise ValueError("sequence must be 1D (flattened) or 2D array")

    elif "features" in data_json:
        feat = np.array(data_json["features"], dtype=np.float32)

        if feat.size == seq_len * feat_dim:
            seq = feat.reshape(seq_len, feat_dim)

        elif feat.size == feat_dim:
            seq = np.tile(feat, (seq_len, 1))

        else:
            raise ValueError(f"features size {feat.size}, expected {feat_dim} or {seq_len*feat_dim}")

    else:
        raise ValueError("Missing 'sequence' or 'features' field in request.")

    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.half()

    return tensor


KEYPOINTS_ROOT = resource_path("KEYPOINTS")
_keypoints_folder_map = {}
_sign_calibration_cache = {}

_METRIC_OFFSET = {
    "dx_chin": 0,
    "dy_chin": 1,
    "dy_fore": 4,
}


def _canon_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _resolve_expected_label(data: dict) -> str | None:
    if not isinstance(data, dict):
        return None

    expected = data.get("expected")
    if isinstance(expected, str) and expected.strip():
        return expected.strip()

    target = data.get("target") or data.get("label")
    if not isinstance(target, str) or not target.strip():
        return None

    aliases = {
        "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "mom": "mother", "nanay": "mother",
        "dad": "father", "tatay": "father",
        "lola": "grandmother", "grandma": "grandmother",
        "lolo": "grandfather", "grandpa": "grandfather",
        "dontunderstand": "understand",
    }
    needle = _canon_name(target)
    needle = aliases.get(needle, needle)

    for class_label in CLASSES:
        cls_key = _canon_name(class_label)
        if cls_key == needle:
            return class_label
        parts = class_label.split("_", 1)
        if len(parts) == 2 and _canon_name(parts[1]) == needle:
            return class_label

    return None


def _init_keypoints_folder_map():
    _keypoints_folder_map.clear()
    if not os.path.isdir(KEYPOINTS_ROOT):
        return
    for name in os.listdir(KEYPOINTS_ROOT):
        folder = os.path.join(KEYPOINTS_ROOT, name)
        if os.path.isdir(folder):
            _keypoints_folder_map[_canon_name(name)] = folder


def _resolve_sign_folder(expected_label: str):
    if not expected_label:
        return None
    exact = os.path.join(KEYPOINTS_ROOT, expected_label)
    if os.path.isdir(exact):
        return exact
    return _keypoints_folder_map.get(_canon_name(expected_label))


def _extract_metric_values(
    seq: np.ndarray,
    metric: str,
    *,
    use_abs: bool = False,
    hand: str = "both",
) -> np.ndarray:
    if seq.ndim != 2 or seq.shape[1] < 200 or metric not in _METRIC_OFFSET:
        return np.array([], dtype=np.float32)

    derived = seq[:, 126:198]
    flags = seq[:, 198:200]
    offset = _METRIC_OFFSET[metric]
    out = []

    for i in range(seq.shape[0]):
        has_left = flags[i, 0] > 0.5
        has_right = flags[i, 1] > 0.5

        include_left = hand in ("both", "left")
        include_right = hand in ("both", "right")

        if has_left and include_left:
            for j in range(6):
                v = derived[i, j * 6 + offset]
                out.append(abs(v) if use_abs else v)

        if has_right and include_right:
            base = 36
            for j in range(6):
                v = derived[i, base + j * 6 + offset]
                out.append(abs(v) if use_abs else v)

    if not out:
        return np.array([], dtype=np.float32)
    return np.array(out, dtype=np.float32)


def _load_sign_calibration(expected_label: str):
    key = _canon_name(expected_label)
    if key in _sign_calibration_cache:
        return _sign_calibration_cache[key]

    folder = _resolve_sign_folder(expected_label)
    if not folder:
        _sign_calibration_cache[key] = None
        return None

    npy_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".npy")
    ]

    dy_fore_values = []
    dy_chin_values = []
    dy_fore_left_values = []
    dy_fore_right_values = []
    abs_dx_values = []
    dx_signed_values = []
    abs_dx_left_values = []
    abs_dx_right_values = []
    dx_signed_left_values = []
    dx_signed_right_values = []
    left_presence = []
    right_presence = []

    for path in npy_files:
        try:
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] < 200:
                continue

            dy_fore = _extract_metric_values(arr, "dy_fore")
            dy_chin = _extract_metric_values(arr, "dy_chin")
            dy_fore_left = _extract_metric_values(arr, "dy_fore", hand="left")
            dy_fore_right = _extract_metric_values(arr, "dy_fore", hand="right")
            abs_dx = _extract_metric_values(arr, "dx_chin", use_abs=True)
            dx_signed = _extract_metric_values(arr, "dx_chin")
            abs_dx_left = _extract_metric_values(arr, "dx_chin", use_abs=True, hand="left")
            abs_dx_right = _extract_metric_values(arr, "dx_chin", use_abs=True, hand="right")
            dx_signed_left = _extract_metric_values(arr, "dx_chin", hand="left")
            dx_signed_right = _extract_metric_values(arr, "dx_chin", hand="right")

            if dy_fore.size:
                dy_fore_values.append(dy_fore)
            if dy_chin.size:
                dy_chin_values.append(dy_chin)
            if dy_fore_left.size:
                dy_fore_left_values.append(dy_fore_left)
            if dy_fore_right.size:
                dy_fore_right_values.append(dy_fore_right)
            if abs_dx.size:
                abs_dx_values.append(abs_dx)
            if dx_signed.size:
                dx_signed_values.append(dx_signed)
            if abs_dx_left.size:
                abs_dx_left_values.append(abs_dx_left)
            if abs_dx_right.size:
                abs_dx_right_values.append(abs_dx_right)
            if dx_signed_left.size:
                dx_signed_left_values.append(dx_signed_left)
            if dx_signed_right.size:
                dx_signed_right_values.append(dx_signed_right)

            flags = arr[:, 198:200]
            left_presence.append(float(np.mean(flags[:, 0] > 0.5)))
            right_presence.append(float(np.mean(flags[:, 1] > 0.5)))
        except Exception:
            continue

    if not dy_fore_values:
        _sign_calibration_cache[key] = None
        return None

    def _make_band(values: np.ndarray, margin: float):
        if values.size == 0:
            return None
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        return {
            "low": float(q25 - margin),
            "high": float(q75 + margin),
            "center": float(q50),
        }

    dy_fore_merged = np.concatenate(dy_fore_values)
    dy_chin_merged = np.concatenate(dy_chin_values) if dy_chin_values else np.array([], dtype=np.float32)
    dy_fore_left_merged = np.concatenate(dy_fore_left_values) if dy_fore_left_values else np.array([], dtype=np.float32)
    dy_fore_right_merged = np.concatenate(dy_fore_right_values) if dy_fore_right_values else np.array([], dtype=np.float32)
    dx_merged = np.concatenate(abs_dx_values) if abs_dx_values else np.array([], dtype=np.float32)
    dx_signed_merged = np.concatenate(dx_signed_values) if dx_signed_values else np.array([], dtype=np.float32)
    dx_left_merged = np.concatenate(abs_dx_left_values) if abs_dx_left_values else np.array([], dtype=np.float32)
    dx_right_merged = np.concatenate(abs_dx_right_values) if abs_dx_right_values else np.array([], dtype=np.float32)
    dx_signed_left_merged = np.concatenate(dx_signed_left_values) if dx_signed_left_values else np.array([], dtype=np.float32)
    dx_signed_right_merged = np.concatenate(dx_signed_right_values) if dx_signed_right_values else np.array([], dtype=np.float32)

    calibration = {
        "label": expected_label,
        "dy_fore": _make_band(dy_fore_merged, 0.03),
        "dy_chin": _make_band(dy_chin_merged, 0.03),
        "dy_fore_left": _make_band(dy_fore_left_merged, 0.03),
        "dy_fore_right": _make_band(dy_fore_right_merged, 0.03),
        "abs_dx_chin": _make_band(dx_merged, 0.02) if dx_merged.size else None,
        "dx_chin_signed": _make_band(dx_signed_merged, 0.02) if dx_signed_merged.size else None,
        "abs_dx_chin_left": _make_band(dx_left_merged, 0.02),
        "abs_dx_chin_right": _make_band(dx_right_merged, 0.02),
        "dx_chin_signed_left": _make_band(dx_signed_left_merged, 0.02),
        "dx_chin_signed_right": _make_band(dx_signed_right_merged, 0.02),
        "left_presence": float(np.mean(left_presence)) if left_presence else 0.0,
        "right_presence": float(np.mean(right_presence)) if right_presence else 0.0,
        "samples": int(dy_fore_merged.size),
    }
    _sign_calibration_cache[key] = calibration
    return calibration


def _build_corrective_feedback(live_seq: np.ndarray, expected_label: str):
    if live_seq.ndim != 2:
        return None

    calibration = _load_sign_calibration(expected_label)
    if not calibration:
        return None

    if live_seq.shape[1] < 200:
        return None

    live_flags = live_seq[:, 198:200]
    live_left_presence = float(np.mean(live_flags[:, 0] > 0.5))
    live_right_presence = float(np.mean(live_flags[:, 1] > 0.5))

    live_dy_fore = _extract_metric_values(live_seq, "dy_fore")
    live_dy_chin = _extract_metric_values(live_seq, "dy_chin")
    live_dy_fore_left = _extract_metric_values(live_seq, "dy_fore", hand="left")
    live_dy_fore_right = _extract_metric_values(live_seq, "dy_fore", hand="right")
    live_abs_dx = _extract_metric_values(live_seq, "dx_chin", use_abs=True)
    live_dx_signed = _extract_metric_values(live_seq, "dx_chin")
    live_abs_dx_left = _extract_metric_values(live_seq, "dx_chin", use_abs=True, hand="left")
    live_abs_dx_right = _extract_metric_values(live_seq, "dx_chin", use_abs=True, hand="right")
    live_dx_signed_left = _extract_metric_values(live_seq, "dx_chin", hand="left")
    live_dx_signed_right = _extract_metric_values(live_seq, "dx_chin", hand="right")

    if live_dy_fore.size < 6:
        return {
            "available": True,
            "source": "npy",
            "status": "insufficient_live_data",
            "message": "Keep your signing hand visible for clearer corrective feedback.",
        }

    cues = []
    status = "good"

    def _band_delta(value: float, band: dict):
        if value > band["high"]:
            return value - band["high"], "high"
        if value < band["low"]:
            return band["low"] - value, "low"
        return 0.0, "inside"

    def _pick_side_delta(left_values: np.ndarray, right_values: np.ndarray, left_band: dict, right_band: dict):
        best = None
        if left_band and left_values.size >= 3:
            left_median = float(np.median(left_values))
            left_delta, left_side = _band_delta(left_median, left_band)
            best = {
                "name": "left",
                "delta": left_delta,
                "side": left_side,
                "median": left_median,
            }
        if right_band and right_values.size >= 3:
            right_median = float(np.median(right_values))
            right_delta, right_side = _band_delta(right_median, right_band)
            if (best is None) or (right_delta > best["delta"]):
                best = {
                    "name": "right",
                    "delta": right_delta,
                    "side": right_side,
                    "median": right_median,
                }
        return best

    if calibration["left_presence"] > 0.65 and live_left_presence < 0.35:
        cues.append((2.5, "Keep your left signing hand visible to the camera."))
    if calibration["right_presence"] > 0.65 and live_right_presence < 0.35:
        cues.append((2.5, "Keep your right signing hand visible to the camera."))

    dy_band = calibration["dy_fore"]
    dy_chin_band = calibration.get("dy_chin")
    dy_left_band = calibration.get("dy_fore_left") or dy_band
    dy_right_band = calibration.get("dy_fore_right") or dy_band
    dy_live = float(np.median(live_dy_fore))
    dy_span = max(1e-3, dy_band["high"] - dy_band["low"])

    height_pick = _pick_side_delta(live_dy_fore_left, live_dy_fore_right, dy_left_band, dy_right_band)
    if height_pick and height_pick["delta"] > 0:
        hand_text = "left hand" if height_pick["name"] == "left" else "right hand"
        side_band = dy_left_band if height_pick["name"] == "left" else dy_right_band
        span = max(1e-3, side_band["high"] - side_band["low"])
        severity = height_pick["delta"] / span
        if height_pick["side"] == "high":
            status = "too_low"
            msg = f"Raise your {hand_text} higher toward forehead level."
            if severity > 1.7:
                msg = f"Raise your {hand_text} much higher toward forehead level."
            cues.append((2.2 + severity, msg))
        elif height_pick["side"] == "low":
            status = "too_high"
            msg = f"Lower your {hand_text} slightly toward forehead level."
            if severity > 1.7:
                msg = f"Lower your {hand_text} more toward forehead level."
            cues.append((2.2 + severity, msg))

    dx_band = calibration.get("abs_dx_chin")
    dx_signed_band = calibration.get("dx_chin_signed")
    dx_left_band = calibration.get("abs_dx_chin_left") or dx_band
    dx_right_band = calibration.get("abs_dx_chin_right") or dx_band
    dx_signed_left_band = calibration.get("dx_chin_signed_left") or dx_signed_band
    dx_signed_right_band = calibration.get("dx_chin_signed_right") or dx_signed_band
    dx_live = None
    if dx_band and live_abs_dx.size >= 6:
        dx_live = float(np.median(live_abs_dx))

    center_pick = _pick_side_delta(live_abs_dx_left, live_abs_dx_right, dx_left_band, dx_right_band)
    if center_pick and center_pick["delta"] > 0:
        hand_text = "left hand" if center_pick["name"] == "left" else "right hand"
        side_band = dx_left_band if center_pick["name"] == "left" else dx_right_band
        span = max(1e-3, side_band["high"] - side_band["low"])
        sev = center_pick["delta"] / span
        if center_pick["side"] == "high":
            cues.append((1.4 + sev, f"Move your {hand_text} closer to the center of your face."))
        elif center_pick["side"] == "low":
            cues.append((1.2 + sev, f"Move your {hand_text} slightly farther from the center of your face."))

    is_number_sign = str(expected_label).lower().startswith("numbers_")
    if is_number_sign:
        number_hand_pick = _pick_side_delta(live_dy_fore_left, live_dy_fore_right, dy_left_band, dy_right_band)
        hand_text = "right hand"
        if number_hand_pick and number_hand_pick.get("name") in ("left", "right"):
            hand_text = f"{number_hand_pick['name']} hand"

        if dy_chin_band and live_dy_chin.size >= 6:
            dy_chin_live = float(np.median(live_dy_chin))
            chin_delta, chin_side = _band_delta(dy_chin_live, dy_chin_band)
            chin_span = max(1e-3, dy_chin_band["high"] - dy_chin_band["low"])
            chin_sev = chin_delta / chin_span
            if chin_side == "low":
                cues.append((3.2 + chin_sev, f"For number signs, keep your {hand_text} below your chin."))
            elif chin_side == "high":
                cues.append((2.3 + chin_sev, f"For number signs, raise your {hand_text} slightly closer under your chin."))

        number_side_pick = _pick_side_delta(live_dx_signed_left, live_dx_signed_right, dx_signed_left_band, dx_signed_right_band)
        if number_side_pick and number_side_pick["delta"] > 0:
            side_band = dx_signed_left_band if number_side_pick["name"] == "left" else dx_signed_right_band
            target_side = "right" if side_band and side_band["center"] >= 0 else "left"
            cues.append((3.0 + number_side_pick["delta"], f"For number signs, position your {hand_text} on the {target_side} side of your face."))

    cues.sort(key=lambda item: item[0], reverse=True)

    if not cues:
        msg = "Good form. Keep the same hand height and position."
    else:
        unique_msgs = []
        for _, text in cues:
            if text not in unique_msgs:
                unique_msgs.append(text)
            if len(unique_msgs) >= 2:
                break
        msg = " ".join(unique_msgs)
        if status == "good":
            status = "adjust"

    return {
        "available": True,
        "source": "npy",
        "status": status,
        "message": msg,
        "live_median_dy_fore": dy_live,
        "target_low_dy_fore": dy_band["low"],
        "target_high_dy_fore": dy_band["high"],
        "target_center_dy_fore": dy_band["center"],
        "live_median_abs_dx_chin": dx_live,
        "expected_left_presence": calibration["left_presence"],
        "expected_right_presence": calibration["right_presence"],
        "live_left_presence": live_left_presence,
        "live_right_presence": live_right_presence,
    }

def get_demo_video_path(label):
    parts = label.split("_")
    if len(parts) != 2:
        return None

    category = parts[0].lower()
    name = parts[1].lower().replace("'", "")

    folder_abs = resource_path(os.path.join("static", "video", category))
    if not os.path.exists(folder_abs):
        return None

    files = os.listdir(folder_abs)
    candidates = [f for f in files if f.lower().startswith(name)]
    if not candidates:
        return None

    chosen = random.choice(candidates)
    return f"static/video/{category}/{chosen}"


_init_keypoints_folder_map()

def run_inference(x):
    """Run inference via TensorRT (if available) or PyTorch."""
    if use_trt:
        # Cast input to whatever dtype the engine actually expects
        input_data = x.cpu().numpy().astype(trt_input_dtype)

        cuda.memcpy_htod_async(d_input, input_data, trt_stream)

        trt_context.set_tensor_address(trt_input_name, int(d_input))
        trt_context.set_tensor_address(trt_output_name, int(d_output))

        trt_context.execute_async_v3(stream_handle=trt_stream.handle)

        output_data = np.empty((1, len(CLASSES)), dtype=trt_output_dtype)
        cuda.memcpy_dtoh_async(output_data, d_output, trt_stream)
        trt_stream.synchronize()

        # Engine outputs raw logits → softmax in FP32
        probs = torch.softmax(
            torch.tensor(output_data.astype(np.float32)),
            dim=1
        ).numpy()[0]

    else:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def log_top3(probs, tag="INFERENCE"): # Print top 3 predictions for debugging
    top3_idx = np.argsort(probs)[::-1][:3]
    print(f"[{tag}] Top-3 predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        print(f"  {rank}. {CLASSES[idx]:.<30s} {probs[idx]*100:6.2f}%")

@app.route("/")
def launch():
    return render_template("splash.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/auto')
@login_required
def auto_recognition():
    return render_template('auto.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/manual')
def manual():
    return render_template("manual.html")

@app.route('/activity')
@login_required
def activity():
    return render_template("activity.html")

@app.route('/detect')
def detect():
    return render_template("detect.html")

@app.route('/results')
@login_required
def results():
    # All individual rows (no limit – full history for PDF export)
    rows = (PracticeResult.query
            .filter_by(user_id=session['user_id'])
            .order_by(PracticeResult.created_at.desc())
            .all())

    results_data = [
        {
            "label": r.label,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in rows
    ]

    daily = (db.session.query( # Daily summaries
                func.date(PracticeResult.created_at).label("day"),
                func.count(PracticeResult.id).label("count")
            )
            .filter(PracticeResult.user_id == session["user_id"])
            .group_by(func.date(PracticeResult.created_at))
            .order_by(func.date(PracticeResult.created_at).desc())
            .all())

    daily_counts = [{"day": (d.day.isoformat() if hasattr(d.day, 'isoformat') else d.day) if d.day else None, "count": int(d.count)} for d in daily]

    top = (db.session.query(PracticeResult.label, func.count(PracticeResult.id).label("c"))
           .filter(PracticeResult.user_id == session["user_id"])
           .group_by(PracticeResult.label)
           .order_by(func.count(PracticeResult.id).desc())
           .first())
    most_common = {"label": top[0], "count": int(top[1])} if top else None

    today = datetime.date.today()
    week_start = today - datetime.timedelta(days=6)

    week_count = (db.session.query(func.count(PracticeResult.id))
                  .filter(PracticeResult.user_id == session["user_id"])
                  .filter(PracticeResult.created_at >= week_start)
                  .scalar()) or 0

    today_count = (db.session.query(func.count(PracticeResult.id))
                   .filter(PracticeResult.user_id == session["user_id"])
                   .filter(func.date(PracticeResult.created_at) == today)
                   .scalar()) or 0

    days_with_activity = {datetime.date.fromisoformat(x["day"]) for x in daily_counts if x["day"]}
    streak = 0
    cursor = today
    while cursor in days_with_activity:
        streak += 1
        cursor -= datetime.timedelta(days=1)

    summary = {
        "today": int(today_count),
        "last_7_days": int(week_count),
        "streak_days": int(streak),
        "most_common": most_common,
    }

    return render_template(
        "results.html",
        db_results_json=json.dumps(results_data),
        results=results_data,
        summary=summary,
        daily_counts=daily_counts,
    )

@app.route('/tutor')
@login_required
def tutor():
    return render_template("tutor.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm = request.form.get('confirm_password') or ''

        if not username or not password:
            flash("Please enter a username and password.", "danger")
            return redirect(url_for('signup'))

        if confirm and password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose another.", "warning")
            return redirect(url_for('signup'))

        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        flash("Account created! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Logged in successfully!", "success")
            return redirect(url_for('home'))

        flash("Invalid username or password.", "danger")
        return redirect(url_for('login'))

    return render_template("login.html")

@app.route('/select')
@login_required
def select():
    return render_template("select.html")

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route("/api/save_result", methods=["POST"])
@login_required
def api_save_result():
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    confidence = data.get("confidence", None)

    if not label:
        return jsonify({"error": "Missing label"}), 400

    save_progress(label, confidence)
    return jsonify({"status": "ok"})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Backend is reachable"})

@app.route("/api/landmarks", methods=["POST"])
def api_landmarks():
    """Accept a raw JPEG frame, run MediaPipe Holistic, return landmarks."""
    img_bytes = request.get_data()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with _mp_lock:
        global _face_frame_counter, _face_count_cache
        results = server_holistic.process(frame_rgb)

        # Face detection every 10th frame to save CPU
        _face_frame_counter += 1
        if _face_frame_counter >= 10:
            _face_frame_counter = 0
            face_results = server_face_detector.process(frame_rgb)
            _face_count_cache = len(face_results.detections) if face_results.detections else 0

    return jsonify({
        "poseLandmarks":     _serialize_landmarks(results.pose_landmarks),
        "faceLandmarks":     _serialize_landmarks(results.face_landmarks),
        "rightHandLandmarks": _serialize_landmarks(results.right_hand_landmarks),
        "leftHandLandmarks":  _serialize_landmarks(results.left_hand_landmarks),
        "personCount":       _face_count_cache,
    })

_GHOST_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghost_cache.json")

def _load_ghost_cache():
    """Load persistent ghost cache from disk on startup."""
    if os.path.exists(_GHOST_CACHE_FILE):
        try:
            with open(_GHOST_CACHE_FILE, "r") as f:
                raw = json.load(f)
            # JSON keys are strings; convert back to (path, frames) tuples
            cache = {}
            for key_str, frames in raw.items():
                parts = key_str.rsplit("|", 1)
                cache[(parts[0], int(parts[1]))] = frames
            print(f"[GHOST] Loaded {len(cache)} cached sign(s) from disk")
            return cache
        except Exception as e:
            print(f"[GHOST] Cache file corrupt, starting fresh: {e}")
    return {}

def _save_ghost_cache():
    """Persist current ghost cache to disk."""
    try:
        # Convert tuple keys to strings for JSON
        raw = {f"{k[0]}|{k[1]}": v for k, v in _ghost_cache.items()}
        with open(_GHOST_CACHE_FILE, "w") as f:
            json.dump(raw, f)
    except Exception as e:
        print(f"[GHOST] Failed to save cache: {e}")

_ghost_cache = _load_ghost_cache()

@app.route("/api/ghost_landmarks", methods=["POST"])
def api_ghost_landmarks():
    """Process a demo video server-side and return landmark frames (cached)."""
    data = request.get_json(force=True)
    video_path = data.get("video_path", "")
    total_frames = int(data.get("total_frames", 24))

    cache_key = (video_path, total_frames)
    if cache_key in _ghost_cache:
        return jsonify({"frames": _ghost_cache[cache_key]})

    # Resolve to absolute path
    abs_path = resource_path(video_path)
    if not os.path.exists(abs_path):
        return jsonify({"error": "Video not found", "path": abs_path}), 404

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0

    frames = []
    for i in range(total_frames):
        ts_ms = (duration * i / total_frames) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use dedicated ghost holistic (static_image_mode) — no lock
        # contention with live camera, and each frame detected independently
        with _ghost_lock:
            results = _ghost_holistic.process(frame_rgb)

        frames.append({
            "poseLandmarks":     _serialize_landmarks(results.pose_landmarks),
            "faceLandmarks":     _serialize_landmarks(results.face_landmarks),
            "rightHandLandmarks": _serialize_landmarks(results.right_hand_landmarks),
            "leftHandLandmarks":  _serialize_landmarks(results.left_hand_landmarks),
        })

    cap.release()
    _ghost_cache[cache_key] = frames  # cache for future requests
    _save_ghost_cache()               # persist to disk
    print(f"[GHOST] Processed & cached {len(frames)} frames for {video_path}")
    return jsonify({"frames": frames})

# Activity Section
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        expected_label = _resolve_expected_label(data)
        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        elif "features" in data:
            x = prepare_sequence({"features": data["features"]})
        else:
            raise ValueError("Missing 'sequence' or 'features'")

        # Save live sequence for debugging comparison
        live_seq = x.squeeze(0).cpu().float().numpy()  # (seq_len, feat_dim)
        np.save("tmp_live_seq.npy", live_seq)
        print(f"[DEBUG] Saved live sequence to tmp_live_seq.npy  shape={live_seq.shape}")

        corrective_feedback = None
        if expected_label:
            corrective_feedback = _build_corrective_feedback(live_seq, expected_label)

        probs = run_inference(x)
        log_top3(probs, tag="PREDICT")
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        
        conf = float(np.max(probs))
        
        NOT_FSL_THRESHOLD = 0.90

        if conf < NOT_FSL_THRESHOLD:
            print(f"[PREDICT] Unrecognized Sign (max_conf={conf:.4f})")
            payload = {
                "prediction": "Unrecognized Sign",
                "confidence": conf,
                "message": "Unrecognized sign"
            }
            if corrective_feedback:
                payload["corrective_feedback"] = corrective_feedback
            return jsonify(payload)

        save_progress(label, conf)

        demo_path = get_demo_video_path(label)
        response = {
            "prediction": label,
            "confidence": conf,
            "demo": demo_path or f"No demo found for {label}"
        }
        if corrective_feedback:
            response["corrective_feedback"] = corrective_feedback
        print(f"[PREDICT] {label} (conf={conf:.4f}) → {demo_path}")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# Auto section
@app.route("/predict_auto", methods=["POST"])
def predict_auto():
    try:
        data = request.get_json(force=True) or {}
        # For error handling of no hand detected
        if (
            ("sequence" not in data and "features" not in data) or
            ("sequence" in data and not data["sequence"]) or
            ("features" in data and not data["features"])
        ):
            return jsonify({
                "prediction": "No hands detected",
                "message": "No hands detected"
            })

        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        else:
            x = prepare_sequence({"features": data["features"]})

        live_seq = x.squeeze(0).cpu().float().numpy()

        probs = run_inference(x)
        log_top3(probs, tag="AUTO")

        conf = float(np.max(probs)) 
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]

        NOT_FSL_THRESHOLD = 0.70
        THRESHOLD = 0.92

        if conf < NOT_FSL_THRESHOLD:
            print(f"[AUTO] Unrecognized Sign (max_conf={conf:.4f})")
            corrective_feedback = _build_corrective_feedback(live_seq, label)
            payload = {
                "prediction": "Unrecognized Sign",
                "confidence": conf,
                "message": "Unrecognized sign"
            }
            if corrective_feedback:
                payload["corrective_feedback"] = corrective_feedback
            return jsonify(payload)

        if conf < THRESHOLD:
            sorted_indices = np.argsort(probs)[::-1]
            top_idx = sorted_indices[0]
            closest_label = CLASSES[top_idx]
            closest_conf = float(probs[top_idx])

            save_progress(closest_label, closest_conf)

            corrective_feedback = _build_corrective_feedback(live_seq, closest_label)

            payload = {
                "prediction": "Incorrect",
                "closest_sign": closest_label,
                "closest_confidence": round(closest_conf, 4),
                "confidence": conf,
                "message": f"Incorrect — closest sign is {closest_label.replace('_', ' ')}"
            }
            if corrective_feedback:
                payload["corrective_feedback"] = corrective_feedback
            return jsonify(payload)

        else:
            save_progress(label, conf)

            corrective_feedback = _build_corrective_feedback(live_seq, label)

            payload = {
                "prediction": label,
                "confidence": conf,
                "message": f"Correct — {label.replace('_', ' ')}"
            }
            if corrective_feedback:
                payload["corrective_feedback"] = corrective_feedback
            return jsonify(payload)

    except Exception as e:
        print(f"[ERROR] Auto Prediction failed: {e}")
        return jsonify({
            "prediction": "Error",
            "message": "Prediction error"
        }), 400

@app.route("/api/assess", methods=["POST"])
def assess():
    try:
        data = request.get_json(force=True)
        x = prepare_sequence(data)
        live_seq = x.squeeze(0).cpu().float().numpy()
        expected_label = _resolve_expected_label(data)

        probs = run_inference(x)
        log_top3(probs, tag="ASSESS")
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]

        save_progress(label, float(np.max(probs)))

        demo_path = get_demo_video_path(label)
        payload = {
            "label": label,
            "probabilities": probs.tolist(),
            "demo": demo_path
        }

        if expected_label:
            corrective_feedback = _build_corrective_feedback(live_seq, expected_label)
            if corrective_feedback:
                payload["corrective_feedback"] = corrective_feedback

        return jsonify(payload)
    except Exception as e:
        print(f"[ERROR] Assessment failed: {e}")
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
