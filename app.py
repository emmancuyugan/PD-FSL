from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from functools import lru_cache

import torch
import torch.nn as nn
import numpy as np
import os
import random
import json
import datetime

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
    """Create tables if they don't exist (safe to call repeatedly)."""
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


# ======================================================
# Inference helper — shared by all prediction routes
# ======================================================
def _run_inference(data_json):
    """Run model inference and return (label, confidence, probs_array).
    Uses torch.inference_mode() for ~5-10% faster inference than no_grad."""
    x = prepare_sequence(data_json)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
    return label, conf, probs


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

MODEL_PATH = r"run2.pt"
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
SEQ_LEN = config.get("SEQ_LEN", 48)

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

print("Loaded Config:")
print("Hidden:", HIDDEN_SIZE)
print("Layers:", NUM_LAYERS)
print("Dropout:", DROPOUT)
print("Classes:", len(CLASSES))
print("SEQ_LEN from config:", SEQ_LEN)
print(f"[APP] Loaded model → hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT}")

def prepare_sequence(data_json):
    seq_len = SEQ_LEN
    feat_dim = INPUT_SIZE

    if "sequence" in data_json:
        seq = np.array(data_json["sequence"], dtype=np.float32)

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

# ======================================================
# Helper — locate demo video automatically
# ======================================================
@lru_cache(maxsize=128)
def _list_video_dir(folder_abs):
    """Cache directory listings to avoid repeated os.listdir calls."""
    if not os.path.exists(folder_abs):
        return ()
    return tuple(os.listdir(folder_abs))


def get_demo_video_path(label):
    parts = label.split("_")
    if len(parts) != 2:
        return None

    category = parts[0].lower()
    name = parts[1].lower().replace("'", "")

    folder_abs = resource_path(os.path.join("static", "video", category))
    files = _list_video_dir(folder_abs)
    if not files:
        return None

    candidates = [f for f in files if f.lower().startswith(name)]
    if not candidates:
        return None

    chosen = random.choice(candidates)
    return f"static/video/{category}/{chosen}"

# ======================================================
# ROUTES — Frontend Pages
# ======================================================
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/vrm-live")
def vrm_live():
    return render_template("vrm-live.html")

@app.route('/auto')
@login_required
def auto_recognition():
    return render_template('auto.html')

@app.route('/about')
def about():
    return render_template("about.html")

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
    # Latest individual rows (for the detailed list/table)
    rows = (PracticeResult.query
            .filter_by(user_id=session['user_id'])
            .order_by(PracticeResult.created_at.desc())
            .limit(500)
            .all())

    results_data = [
        {
            "label": r.label,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in rows
    ]

    # ---- Daily summaries ----
    daily = (db.session.query(
                func.date(PracticeResult.created_at).label("day"),
                func.count(PracticeResult.id).label("count")
            )
            .filter(PracticeResult.user_id == session["user_id"])
            .group_by(func.date(PracticeResult.created_at))
            .order_by(func.date(PracticeResult.created_at).desc())
            .all())

    daily_counts = [{"day": d.day.isoformat() if d.day else None, "count": int(d.count)} for d in daily]

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

# ======================================================
# API Routes (backend logic)
# ======================================================
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
    return jsonify({"message": "Backend is reachable ✅"})

# --------------------------
# Normal /predict (Activity)
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        label, conf, probs = _run_inference(data)

        save_progress(label, conf)

        demo_path = get_demo_video_path(label)
        response = {
            "prediction": label,
            "confidence": conf,
            "demo": demo_path or f"No demo found for {label}"
        }
        print(f"[PREDICT] {label} (conf={conf:.4f}) → {demo_path}")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# --------------------------
# New /predict_auto (Auto Recognition Only)
# --------------------------
@app.route("/predict_auto", methods=["POST"])
def predict_auto():
    try:
        data = request.get_json(force=True)
        label, conf, probs = _run_inference(data)

        THRESHOLD = 0.8
        if conf < THRESHOLD:
            save_progress("Incorrect", conf)

            response = {
                "prediction": "Incorrect",
                "closest_sign": label,
                "closest_confidence": round(conf, 4),
                "confidence": conf,
                "message": f"❌ Incorrect — closest sign you performed is {label.replace('_', ' ')}"
            }
            print(f"[AUTO] Incorrect (conf={conf:.4f}) → Closest: {label} ({conf:.4f})")
        else:
            save_progress(label, conf)

            response = {
                "prediction": label,
                "confidence": conf,
                "message": f"✅ Correct — {label.replace('_', ' ')}"
            }
            print(f"[AUTO] {label} (conf={conf:.4f}) [threshold={THRESHOLD}]")

        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Auto Prediction failed: {e}")
        return jsonify({"error": f"Auto Prediction failed: {str(e)}"}), 400

# --------------------------
# /api/assess
# --------------------------
@app.route("/api/assess", methods=["POST"])
def assess():
    try:
        data = request.get_json(force=True)
        label, conf, probs = _run_inference(data)

        save_progress(label, conf)

        demo_path = get_demo_video_path(label)
        return jsonify({
            "label": label,
            "probabilities": probs.tolist(),
            "demo": demo_path
        })
    except Exception as e:
        print(f"[ERROR] Assessment failed: {e}")
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500


# --------------------------
# /api/random — provide random challenge phrase for detect.html
# --------------------------
@app.route("/api/random", methods=["GET"])
def api_random():
    phrase = random.choice(CLASSES) if CLASSES else "hello"
    return jsonify({"phrase": phrase.replace("_", " ")})

# ======================================================
# Run app
# ======================================================
if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
