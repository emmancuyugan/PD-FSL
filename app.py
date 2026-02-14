from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

import torch
import torch.nn as nn
import numpy as np
import os
import random
import json
import datetime

from model import ModifiedLSTM
from pathutils import resource_path

# ======================================================
# Flask setup
# ======================================================

app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static"),
)

# Optional (kept) – if you need CORS later
CORS(app)

# ======================================================
# Environment + DB setup (offline/local)
# ======================================================
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ======================================================
# Database models
# ======================================================
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

# ======================================================
# Model setup
# ======================================================
MODEL_PATH = resource_path("run35.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Color_Black",
    "Color_Blue",
    "Color_Green",
    "Color_Orange",
    "Color_Pink",
    "Color_Red",
    "Color_White",
    "Color_Yellow",
    "Family_Daughter",
    "Family_Father",
    "Family_Grandfather",
    "Family_Grandmother",
    "Family_Mother",
    "Family_Son",
    "Numbers_Five",
    "Numbers_Four",
    "Numbers_One",
    "Numbers_Three",
    "Numbers_Two",
    "Relationship_Boy",
    "Relationship_Girl",
    "Relationship_Man",
    "Relationship_Woman",
    "Survival_Correct",
    "Survival_Don'tUnderstand",
    "Survival_No",
    "Survival_Understand",
    "Survival_Wrong",
    "Survival_Yes",
]

INPUT_SIZE = 188
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = len(CLASSES)
SEQ_LEN = 48

model = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                     dropout=0.35, use_layernorm=True).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ======================================================
# ✅ prepare_sequence (unchanged)
# ======================================================
def prepare_sequence(data_json):
    SEQ_LEN, FEAT_DIM = 48, 188
    if "sequence" in data_json:
        seq = np.array(data_json["sequence"], dtype=np.float32)
        if seq.ndim == 1 and seq.size == SEQ_LEN * FEAT_DIM:
            seq = seq.reshape(SEQ_LEN, FEAT_DIM)
        elif seq.ndim == 2:
            if seq.shape != (SEQ_LEN, FEAT_DIM):
                raise ValueError(f"sequence shape {seq.shape}, expected {(SEQ_LEN, FEAT_DIM)}")
        else:
            raise ValueError("sequence must be 1D (flattened) or 2D array")
    elif "features" in data_json:
        feat = np.array(data_json["features"], dtype=np.float32)
        if feat.size == SEQ_LEN * FEAT_DIM:
            seq = feat.reshape(SEQ_LEN, FEAT_DIM)
        elif feat.size == FEAT_DIM:
            seq = np.tile(feat, (SEQ_LEN, 1))
        else:
            raise ValueError(f"features size {feat.size}, expected {FEAT_DIM} or {SEQ_LEN*FEAT_DIM}")
    else:
        raise ValueError("Missing 'sequence' or 'features' field in request.")
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

# ======================================================
# Helper — locate demo video automatically
# ======================================================
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
            return redirect(url_for('select'))

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
        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        elif "features" in data:
            x = prepare_sequence({"features": data["features"]})
        else:
            raise ValueError("Missing 'sequence' or 'features'")

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        conf = float(np.max(probs))
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

        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        elif "features" in data:
            x = prepare_sequence({"features": data["features"]})
        else:
            raise ValueError("Missing 'sequence' or 'features'")

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            conf = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        THRESHOLD = 0.8
        if conf < THRESHOLD:
            sorted_indices = np.argsort(probs)[::-1]
            top_idx = sorted_indices[0]
            closest_label = CLASSES[top_idx]
            closest_conf = float(probs[top_idx])

            # Save "closest" (what the model thinks you performed)
            save_progress(closest_label, closest_conf)

            response = {
                "prediction": "Incorrect",
                "closest_sign": closest_label,
                "closest_confidence": round(closest_conf, 4),
                "confidence": conf,
                "message": f"❌ Incorrect — closest sign you performed is {closest_label.replace('_', ' ')}"
            }
            print(f"[AUTO] Incorrect (conf={conf:.4f}) → Closest: {closest_label} ({closest_conf:.4f})")
        else:
            # Save correct sign
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
        x = prepare_sequence(data)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        save_progress(label, float(np.max(probs)))

        demo_path = get_demo_video_path(label)
        return jsonify({
            "label": label,
            "probabilities": probs.tolist(),
            "demo": demo_path
        })
    except Exception as e:
        print(f"[ERROR] Assessment failed: {e}")
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500

# ======================================================
# Run app
# ======================================================
if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
