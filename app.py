from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

# Libraries for model inference and MediaPipe
import torch
import numpy as np
import os
import pathlib, platform
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PurePosixPath
import random
import json
import datetime
import threading
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
MODEL_A_PATH = os.getenv("MODEL_A_PATH", r"run57.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def _build_model_from_checkpoint(model_path):
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]

    classes = config["CLASSES"]
    input_size = config["FEATURE_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    dropout = config["DROPOUT"]
    seq_len = config.get("SEQ_LEN", config.get("SEQUENCE_LENGTH", 48))

    loaded_model = ModifiedLSTM(
        input_size,
        hidden_size,
        num_layers,
        len(classes),
        dropout=dropout,
        use_layernorm=True,
    ).to(device)

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)

    if device.type == "cuda":
        loaded_model.half()

    loaded_model.eval()

    return {
        "path": model_path,
        "classes": classes,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": seq_len,
        "model": loaded_model,
    }


MODEL_PROFILES = {
    "A": _build_model_from_checkpoint(MODEL_A_PATH),
}

INPUT_SIZE = MODEL_PROFILES["A"]["input_size"]
SEQ_LEN = MODEL_PROFILES["A"]["seq_len"]

for model_key, profile in MODEL_PROFILES.items():
    if profile["input_size"] != INPUT_SIZE:
        raise ValueError(
            f"Model {model_key} FEATURE_DIM mismatch: {profile['input_size']} vs {INPUT_SIZE}"
        )
    if profile["seq_len"] != SEQ_LEN:
        raise ValueError(
            f"Model {model_key} SEQ_LEN mismatch: {profile['seq_len']} vs {SEQ_LEN}"
        )


def _normalize_sign_key(raw_label):
    if raw_label is None:
        return ""

    text = str(raw_label).strip().lower()
    if not text:
        return ""

    if "+" in text:
        text = text.split("+", 1)[0].strip()

    if "_" in text:
        text = text.split("_")[-1]

    token = "".join(ch for ch in text if ch.isalnum())

    aliases = {
        "1": "one", "one": "one",
        "2": "two", "two": "two",
        "3": "three", "three": "three",
        "4": "four", "four": "four",
        "5": "five", "five": "five",
        "6": "six", "six": "six",
        "7": "seven", "seven": "seven",
        "8": "eight", "eight": "eight",
        "9": "nine", "nine": "nine",
        "10": "ten", "ten": "ten",
        "11": "eleven", "eleven": "eleven",
        "12": "twelve", "twelve": "twelve",
        "13": "thirteen", "thirteen": "thirteen",
        "14": "fourteen", "fourteen": "fourteen",
        "15": "fifteen", "fifteen": "fifteen",
        "16": "sixteen", "sixteen": "sixteen",
        "17": "seventeen", "seventeen": "seventeen",
        "18": "eighteen", "eighteen": "eighteen",
        "19": "nineteen", "nineteen": "nineteen",
        "20": "twenty", "twenty": "twenty",
        "mother": "mother",
        "father": "father",
        "son": "son",
        "daughter": "daughter",
        "grandfather": "grandfather",
        "grandmother": "grandmother",
        "auntie": "auntie",
        "uncle": "uncle",
        "cousin": "cousin",
        "black": "black",
        "white": "white",
        "pink": "pink",
        "red": "red",
        "yellow": "yellow",
        "blue": "blue",
        "green": "green",
        "orange": "orange",
        "violet": "violet",
        "boy": "boy",
        "girl": "girl",
        "yes": "yes",
        "no": "no",
        "understand": "understand",
        "wrong": "wrong",
        "correct": "correct",
        "please": "please",
        "thankyou": "thankyou",
        "thanks": "thankyou",
        "drink": "drink",
        "eat": "eat",
        "cook": "cook",
        "coffee": "coffee",
        "juice": "juice",
        "meat": "meat",
        "rice": "rice",
        "milk": "milk",
        "eggs": "eggs",
        "fish": "fish",
        "chicken": "chicken",
    }

    return aliases.get(token, token)


def _route_model_for_sign(raw_label):
    # Single-model experiment: route every sign to model A.
    _ = _normalize_sign_key(raw_label)
    return "A"


def _pick_model_for_request(data):
    _ = data
    return "A"


def _pick_model_hint_for_request(data):
    """Single-model experiment: no hint needed."""
    _ = data
    return "A"

# Get tensor dtypes for TensorRT engine if available
TRT_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "model.engine")
use_trt = False
TRT_MODEL_KEY = "A"

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
    output_shape = (1, len(MODEL_PROFILES[TRT_MODEL_KEY]["classes"]))

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
print("SEQ_LEN from config:", SEQ_LEN)
for model_key, profile in MODEL_PROFILES.items():
    print(
        f"[APP] Loaded model {model_key} from {profile['path']} "
        f"→ hidden={profile['hidden_size']}, layers={profile['num_layers']}, "
        f"dropout={profile['dropout']}, classes={len(profile['classes'])}"
    )

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

def run_inference(x, model_key="A"):
    """Run inference via TensorRT (model A only) or PyTorch for selected model."""
    profile = MODEL_PROFILES[model_key]
    model_classes = profile["classes"]

    if use_trt and model_key == TRT_MODEL_KEY:
        # Cast input to whatever dtype the engine actually expects
        input_data = x.cpu().numpy().astype(trt_input_dtype)

        cuda.memcpy_htod_async(d_input, input_data, trt_stream)

        trt_context.set_tensor_address(trt_input_name, int(d_input))
        trt_context.set_tensor_address(trt_output_name, int(d_output))

        trt_context.execute_async_v3(stream_handle=trt_stream.handle)

        output_data = np.empty((1, len(model_classes)), dtype=trt_output_dtype)
        cuda.memcpy_dtoh_async(output_data, d_output, trt_stream)
        trt_stream.synchronize()

        # Engine outputs raw logits → softmax in FP32
        probs = torch.softmax(
            torch.tensor(output_data.astype(np.float32)),
            dim=1
        ).numpy()[0]

    else:
        with torch.no_grad():
            logits = profile["model"](x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def _normalized_confidence(max_prob, num_classes):
    """Map max softmax to [0,1] while accounting for class-count baseline (1/N)."""
    if num_classes <= 1:
        return float(max_prob)
    baseline = 1.0 / float(num_classes)
    denom = max(1e-8, 1.0 - baseline)
    norm = (float(max_prob) - baseline) / denom
    return max(0.0, min(1.0, norm))


MODEL_SCORE_WEIGHTS = {
    "A": float(os.getenv("AUTO_MODEL_A_WEIGHT", "1.0")),
    "B": float(os.getenv("AUTO_MODEL_B_WEIGHT", "1.0")),
    "C": float(os.getenv("AUTO_MODEL_C_WEIGHT", "0.97")),
}

MODEL_C_OVERRIDE_MARGIN = float(os.getenv("AUTO_MODEL_C_OVERRIDE_MARGIN", "0.04"))
NON_C_STRONG_CONF = float(os.getenv("AUTO_NON_C_STRONG_CONF", "0.985"))
EXPECTED_MODEL_SCORE_BONUS = float(os.getenv("AUTO_EXPECTED_MODEL_SCORE_BONUS", "0.08"))


def log_top3(probs, classes, tag="INFERENCE"): # Print top 3 predictions for debugging
    if not np.any(np.isfinite(probs)):
        print(f"[{tag}] Unrecognized sign (non-finite probabilities)")
        return

    top3_idx = np.argsort(probs)[::-1][:3]
    print(f"[{tag}] Top-3 predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        print(f"  {rank}. {classes[idx]:.<30s} {probs[idx]*100:6.2f}%")

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

        model_key = _pick_model_for_request(data)
        classes = MODEL_PROFILES[model_key]["classes"]

        probs = run_inference(x, model_key=model_key)
        log_top3(probs, classes, tag=f"PREDICT-{model_key}")

        if not np.any(np.isfinite(probs)):
            print("[PREDICT] Unrecognized Sign (non-finite probabilities)")
            return jsonify({
                "prediction": "Unrecognized Sign",
                "confidence": 0.0,
                "message": "Unrecognized sign"
            })

        pred_idx = int(np.argmax(probs))
        label = classes[pred_idx]

        conf = float(np.max(probs))

        NOT_FSL_THRESHOLD = 0.90

        if conf < NOT_FSL_THRESHOLD:
            print(f"[PREDICT] Unrecognized Sign (max_conf={conf:.4f})")
            return jsonify({
                "prediction": "Unrecognized Sign",
                "confidence": conf,
                "message": "Unrecognized sign"
            })

        save_progress(label, conf)

        demo_path = get_demo_video_path(label)
        response = {
            "prediction": label,
            "confidence": conf,
            "model": model_key,
            "demo": demo_path or f"No demo found for {label}"
        }
        print(f"[PREDICT-{model_key}] {label} (conf={conf:.4f}) → {demo_path}")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# Auto section
@app.route("/predict_auto", methods=["POST"])
def predict_auto():
    try:
        data = request.get_json(force=True) or {}
        realtime_probe = bool(data.get("realtime", False))
        expected_model_hint = _pick_model_hint_for_request(data)
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

        model_results = []
        for model_key, profile in MODEL_PROFILES.items():
            probs = run_inference(x, model_key=model_key)
            classes = profile["classes"]
            log_top3(probs, classes, tag=f"AUTO-{model_key}")

            if not np.any(np.isfinite(probs)):
                continue

            pred_idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            norm_conf = _normalized_confidence(conf, len(classes))
            weighted_score = norm_conf * MODEL_SCORE_WEIGHTS.get(model_key, 1.0)
            if expected_model_hint and model_key == expected_model_hint:
                weighted_score += EXPECTED_MODEL_SCORE_BONUS
            model_results.append({
                "model": model_key,
                "probs": probs,
                "classes": classes,
                "pred_idx": pred_idx,
                "label": classes[pred_idx],
                "conf": conf,
                "norm_conf": norm_conf,
                "weighted_score": weighted_score,
            })
            print(
                f"[AUTO-SCORE] model={model_key} label={classes[pred_idx]} "
                f"raw={conf:.4f} norm={norm_conf:.4f} "
                f"weight={MODEL_SCORE_WEIGHTS.get(model_key, 1.0):.3f} "
                f"bonus={EXPECTED_MODEL_SCORE_BONUS if (expected_model_hint and model_key == expected_model_hint) else 0.0:.3f} "
                f"score={weighted_score:.4f}"
            )

        if not model_results:
            print("[AUTO] Unrecognized Sign (non-finite probabilities)")
            return jsonify({
                "prediction": "Unrecognized Sign",
                "confidence": 0.0,
                "message": "Unrecognized sign"
            })

        NOT_FSL_THRESHOLD = 0.70
        THRESHOLD = 0.92
        NOT_FSL_NORM_THRESHOLD = 0.72
        THRESHOLD_NORM = 0.90

        best_any = max(model_results, key=lambda r: r["weighted_score"])
        candidates = [
            r for r in model_results
            if r["conf"] >= NOT_FSL_THRESHOLD and r["norm_conf"] >= NOT_FSL_NORM_THRESHOLD
        ]

        if not candidates:
            print(
                f"[AUTO] Unrecognized Sign (best_model={best_any['model']}, "
                f"raw={best_any['conf']:.4f}, norm={best_any['norm_conf']:.4f})"
            )
            return jsonify({
                "prediction": "Unrecognized Sign",
                "confidence": best_any["conf"],
                "message": "Unrecognized sign"
            })

        best = max(candidates, key=lambda r: r["weighted_score"])

        # Guard against model C dominating due to easier calibration:
        # if C only narrowly beats a very confident A/B, prefer non-C.
        if (
            best["model"] == "C"
            and not expected_model_hint
            and MODEL_C_OVERRIDE_MARGIN > 0.0
        ):
            non_c = [r for r in candidates if r["model"] != "C"]
            if non_c:
                best_non_c = max(non_c, key=lambda r: r["weighted_score"])
                close_score = best_non_c["weighted_score"] >= (best["weighted_score"] - MODEL_C_OVERRIDE_MARGIN)
                strong_non_c = best_non_c["conf"] >= NON_C_STRONG_CONF
                if close_score and strong_non_c:
                    print(
                        f"[AUTO] Overriding C with {best_non_c['model']} "
                        f"(C_score={best['weighted_score']:.4f}, nonC_score={best_non_c['weighted_score']:.4f}, "
                        f"nonC_raw={best_non_c['conf']:.4f})"
                    )
                    best = best_non_c

        conf = best["conf"]
        label = best["label"]
        norm_conf = best["norm_conf"]
        print(
            f"[AUTO-CHOSEN] model={best['model']} label={label} "
            f"raw={conf:.4f} norm={norm_conf:.4f} score={best['weighted_score']:.4f} "
            f"expected_hint={expected_model_hint or 'none'}"
        )

        if conf < THRESHOLD or norm_conf < THRESHOLD_NORM:
            closest_label = best["label"]
            closest_conf = best["conf"]

            if not realtime_probe:
                save_progress(closest_label, closest_conf)

            return jsonify({
                "prediction": "Incorrect",
                "closest_sign": closest_label,
                "closest_confidence": round(closest_conf, 4),
                "normalized_confidence": round(norm_conf, 4),
                "confidence": conf,
                "model": best["model"],
                "message": f"Incorrect — closest sign is {closest_label.replace('_', ' ')}"
            })

        else:
            if not realtime_probe:
                save_progress(label, conf)

            return jsonify({
                "prediction": label,
                "confidence": conf,
                "normalized_confidence": round(norm_conf, 4),
                "model": best["model"],
                "message": f"Correct — {label.replace('_', ' ')}"
            })

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

        model_key = _pick_model_for_request(data)
        classes = MODEL_PROFILES[model_key]["classes"]

        probs = run_inference(x, model_key=model_key)
        log_top3(probs, classes, tag=f"ASSESS-{model_key}")
        pred_idx = int(np.argmax(probs))
        label = classes[pred_idx]

        save_progress(label, float(np.max(probs)))

        demo_path = get_demo_video_path(label)
        return jsonify({
            "label": label,
            "model": model_key,
            "probabilities": probs.tolist(),
            "demo": demo_path
        })
    except Exception as e:
        print(f"[ERROR] Assessment failed: {e}")
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
