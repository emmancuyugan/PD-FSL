import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from model import ModifiedLSTM


def normalize_sign_key(raw_label):
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


MODEL_A_SIGNS = {
    "one", "two", "three", "four", "five",
    "mother", "father", "son", "daughter", "grandfather", "grandmother",
    "black", "white", "pink", "red", "yellow", "blue", "green", "orange",
    "boy", "girl",
    "yes", "no", "understand", "wrong", "correct",
}

MODEL_B_SIGNS = {
    "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "auntie", "uncle", "cousin",
    "violet",
    "please", "thankyou",
}

MODEL_C_SIGNS = {
    "coffee", "juice", "meat", "rice", "milk", "eggs", "fish", "chicken",
}


def route_model_for_sign(sign_key):
    if sign_key in MODEL_A_SIGNS:
        return "A"
    if sign_key in MODEL_B_SIGNS:
        return "B"
    if sign_key in MODEL_C_SIGNS:
        return "C"
    return None


def normalized_confidence(max_prob, num_classes):
    if num_classes <= 1:
        return float(max_prob)
    baseline = 1.0 / float(num_classes)
    denom = max(1e-8, 1.0 - baseline)
    norm = (float(max_prob) - baseline) / denom
    return max(0.0, min(1.0, norm))


def build_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    loaded_model.eval()

    return {
        "path": model_path,
        "classes": classes,
        "input_size": input_size,
        "seq_len": seq_len,
        "model": loaded_model,
    }


def prepare_sequence(seq_raw, seq_len, feat_dim, device):
    seq = np.array(seq_raw, dtype=np.float32)

    if seq.ndim == 1 and seq.size == seq_len * feat_dim:
        seq = seq.reshape(seq_len, feat_dim)
    elif seq.ndim == 2 and seq.shape == (seq_len, feat_dim):
        pass
    elif seq.size == feat_dim:
        seq = np.tile(seq.reshape(1, feat_dim), (seq_len, 1))
    else:
        raise ValueError(f"Unexpected npy shape={seq.shape}, size={seq.size}")

    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)


def infer_scores(x, profiles, weights, biases, expected_hint=None, expected_bonus=0.08):
    out = {}
    for model_key, profile in profiles.items():
        with torch.no_grad():
            logits = profile["model"](x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        norm_conf = normalized_confidence(conf, len(profile["classes"]))
        weighted = norm_conf * weights.get(model_key, 1.0)
        weighted += biases.get(model_key, 0.0)

        if expected_hint and expected_hint == model_key:
            weighted += expected_bonus

        out[model_key] = {
            "label": profile["classes"][pred_idx],
            "conf": conf,
            "norm_conf": norm_conf,
            "weighted_score": weighted,
        }

    return out


def choose_best(scores):
    return max(scores.items(), key=lambda kv: kv[1]["weighted_score"])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--npy-dir", default=os.path.join("Training", "Model npy"))
    parser.add_argument("--a", default="run20.pt")
    parser.add_argument("--b", default="run47.pt")
    parser.add_argument("--c", default="run53.pt")
    parser.add_argument("--weight-a", type=float, default=1.0)
    parser.add_argument("--weight-b", type=float, default=1.0)
    parser.add_argument("--weight-c", type=float, default=1.0)
    parser.add_argument("--bias-a", type=float, default=0.0)
    parser.add_argument("--bias-b", type=float, default=0.0)
    parser.add_argument("--bias-c", type=float, default=0.0)
    parser.add_argument("--expected-bonus", type=float, default=0.08)
    parser.add_argument("--out", default="npy_bias_report.json")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    npy_root = os.path.join(root, args.npy_dir)

    weights = {"A": args.weight_a, "B": args.weight_b, "C": args.weight_c}
    biases = {"A": args.bias_a, "B": args.bias_b, "C": args.bias_c}

    device = torch.device("cpu")

    profiles = {
        "A": build_model_from_checkpoint(os.path.join(root, args.a), device),
        "B": build_model_from_checkpoint(os.path.join(root, args.b), device),
        "C": build_model_from_checkpoint(os.path.join(root, args.c), device),
    }

    seq_len = profiles["A"]["seq_len"]
    feat_dim = profiles["A"]["input_size"]

    per_folder = {}
    agg = {
        "A": {"n": 0, "correct_no_hint": 0, "correct_with_hint": 0},
        "B": {"n": 0, "correct_no_hint": 0, "correct_with_hint": 0},
        "C": {"n": 0, "correct_no_hint": 0, "correct_with_hint": 0},
    }

    bad_shape_files = []

    folders = [
        d for d in os.listdir(npy_root)
        if os.path.isdir(os.path.join(npy_root, d))
    ]

    for folder in sorted(folders):
        folder_path = os.path.join(npy_root, folder)
        npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
        if not npy_files:
            continue

        sign_key = normalize_sign_key(folder)
        expected_model = route_model_for_sign(sign_key)

        counts_no_hint = defaultdict(int)
        counts_with_hint = defaultdict(int)

        for npy_path in npy_files:
            seq_raw = np.load(npy_path)
            try:
                x = prepare_sequence(seq_raw, seq_len, feat_dim, device)
            except Exception as ex:
                bad_shape_files.append({"file": npy_path, "error": str(ex)})
                continue

            scores_no_hint = infer_scores(
                x,
                profiles,
                weights,
                biases,
                expected_hint=None,
                expected_bonus=args.expected_bonus,
            )
            best_no_hint = choose_best(scores_no_hint)
            counts_no_hint[best_no_hint] += 1

            scores_with_hint = infer_scores(
                x,
                profiles,
                weights,
                biases,
                expected_hint=expected_model,
                expected_bonus=args.expected_bonus,
            )
            best_with_hint = choose_best(scores_with_hint)
            counts_with_hint[best_with_hint] += 1

        total_ok = sum(counts_no_hint.values())

        per_folder[folder] = {
            "n_total": len(npy_files),
            "n_scored": total_ok,
            "sign_key": sign_key,
            "expected_model": expected_model,
            "no_hint": {k: int(v) for k, v in counts_no_hint.items()},
            "with_hint": {k: int(v) for k, v in counts_with_hint.items()},
            "no_hint_expected_rate": (
                (counts_no_hint[expected_model] / total_ok) if expected_model and total_ok else None
            ),
            "with_hint_expected_rate": (
                (counts_with_hint[expected_model] / total_ok) if expected_model and total_ok else None
            ),
        }

        if expected_model in agg:
            agg[expected_model]["n"] += total_ok
            agg[expected_model]["correct_no_hint"] += int(counts_no_hint[expected_model])
            agg[expected_model]["correct_with_hint"] += int(counts_with_hint[expected_model])

    for model_key, row in agg.items():
        n = row["n"]
        row["no_hint_rate"] = (row["correct_no_hint"] / n) if n else None
        row["with_hint_rate"] = (row["correct_with_hint"] / n) if n else None

    focus_keys = [
        k for k in per_folder.keys()
        if "additionals_meat" in k.lower()
        or "additionals_juice" in k.lower()
        or "additionals_fish" in k.lower()
        or "additionals_chicken" in k.lower()
        or "numbers_one" in k.lower()
        or "numbers_eleven" in k.lower()
    ]

    report = {
        "weights": weights,
        "biases": biases,
        "expected_bonus": args.expected_bonus,
        "agg": agg,
        "focus": {k: per_folder[k] for k in sorted(focus_keys)},
        "bad_shape_count": len(bad_shape_files),
        "bad_shape_files": bad_shape_files[:25],
    }

    out_path = os.path.join(root, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
