import argparse
import json
import time
from typing import Dict, List, Optional
from urllib import request

import cv2
from pythonosc.udp_client import SimpleUDPClient


Landmark = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge PD-FSL MediaPipe landmarks to OSC for Warudo (or any OSC receiver)."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/api/landmarks", help="PD-FSL landmarks endpoint")
    parser.add_argument("--osc-host", default="127.0.0.1", help="OSC target host (Warudo machine)")
    parser.add_argument("--osc-port", type=int, default=9000, help="OSC target port")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality for API frame upload (1-100)")
    parser.add_argument("--fps", type=float, default=20.0, help="Upload/send frequency limit")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Smoothing alpha (0..1). Lower = smoother, higher = more responsive.",
    )
    parser.add_argument(
        "--osc-prefix",
        default="/pd_fsl",
        help="OSC address prefix. Example outputs: /pd_fsl/left_hand/0",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show local camera preview window (press q to quit).",
    )
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def send_frame_to_api(api_url: str, frame_bgr, jpeg_quality: int) -> Dict:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(clamp(jpeg_quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("Failed to encode webcam frame to JPEG")

    req = request.Request(api_url, data=encoded.tobytes(), method="POST")
    req.add_header("Content-Type", "image/jpeg")

    with request.urlopen(req, timeout=3.0) as resp:
        payload = resp.read().decode("utf-8")

    return json.loads(payload)


def smooth_landmarks(
    previous: Optional[List[Landmark]],
    current: Optional[List[Landmark]],
    alpha: float,
) -> Optional[List[Landmark]]:
    if not current:
        return current
    if not previous or len(previous) != len(current):
        return current

    out: List[Landmark] = []
    for prev_pt, curr_pt in zip(previous, current):
        if not prev_pt:
            out.append(curr_pt)
            continue
        out.append(
            {
                "x": prev_pt["x"] + (curr_pt["x"] - prev_pt["x"]) * alpha,
                "y": prev_pt["y"] + (curr_pt["y"] - prev_pt["y"]) * alpha,
                "z": prev_pt["z"] + (curr_pt["z"] - prev_pt["z"]) * alpha,
                "visibility": prev_pt.get("visibility", 0.0)
                + (curr_pt.get("visibility", 0.0) - prev_pt.get("visibility", 0.0)) * alpha,
            }
        )
    return out


def send_landmark_list(client: SimpleUDPClient, path_prefix: str, points: Optional[List[Landmark]]) -> None:
    if not points:
        return
    for idx, point in enumerate(points):
        client.send_message(
            f"{path_prefix}/{idx}",
            [
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
                float(point.get("visibility", 0.0)),
            ],
        )


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    osc = SimpleUDPClient(args.osc_host, args.osc_port)
    frame_period = 1.0 / max(1.0, args.fps)

    prev = {
        "poseLandmarks": None,
        "faceLandmarks": None,
        "leftHandLandmarks": None,
        "rightHandLandmarks": None,
    }

    print("[Bridge] Starting PD-FSL -> OSC stream")
    print(f"[Bridge] API: {args.api_url}")
    print(f"[Bridge] OSC: {args.osc_host}:{args.osc_port}")
    print(f"[Bridge] Prefix: {args.osc_prefix}")

    last_tick = 0.0
    sent = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if args.show_preview:
                cv2.imshow("PD-FSL Bridge Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_tick < frame_period:
                continue
            last_tick = now

            try:
                payload = send_frame_to_api(args.api_url, frame, args.jpeg_quality)
            except Exception as exc:
                print(f"[Bridge] API error: {exc}")
                continue

            pose = smooth_landmarks(prev["poseLandmarks"], payload.get("poseLandmarks"), args.alpha)
            face = smooth_landmarks(prev["faceLandmarks"], payload.get("faceLandmarks"), args.alpha)
            left_hand = smooth_landmarks(prev["leftHandLandmarks"], payload.get("leftHandLandmarks"), args.alpha)
            right_hand = smooth_landmarks(prev["rightHandLandmarks"], payload.get("rightHandLandmarks"), args.alpha)

            prev["poseLandmarks"] = pose
            prev["faceLandmarks"] = face
            prev["leftHandLandmarks"] = left_hand
            prev["rightHandLandmarks"] = right_hand

            send_landmark_list(osc, f"{args.osc_prefix}/pose", pose)
            send_landmark_list(osc, f"{args.osc_prefix}/face", face)
            send_landmark_list(osc, f"{args.osc_prefix}/left_hand", left_hand)
            send_landmark_list(osc, f"{args.osc_prefix}/right_hand", right_hand)

            osc.send_message(f"{args.osc_prefix}/meta/person_count", int(payload.get("personCount") or 0))
            osc.send_message(f"{args.osc_prefix}/meta/timestamp", float(time.time()))

            sent += 1
            if sent % 50 == 0:
                elapsed = max(1e-6, time.time() - start)
                print(f"[Bridge] Sent {sent} landmark packets ({sent / elapsed:.1f} fps effective)")

    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
