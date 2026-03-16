# Filipino-Sign-Language
Updated FSL project for Design Project that doesn't need internet connectivity.

## Warudo OSC Bridge (using PD-FSL MediaPipe)

You can stream landmarks from the existing `/api/landmarks` endpoint to Warudo (or any OSC receiver).

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start PD-FSL app

Run your normal app startup so `/api/landmarks` is available (default `http://127.0.0.1:5000/api/landmarks`).

### 3) Run bridge

```bash
python warudo_osc_bridge.py --osc-host 127.0.0.1 --osc-port 9000 --fps 20 --alpha 0.35
```

Optional preview window:

```bash
python warudo_osc_bridge.py --show-preview
```

### OSC addresses emitted

- `/pd_fsl/pose/{index}` -> `[x, y, z, visibility]`
- `/pd_fsl/face/{index}` -> `[x, y, z, visibility]`
- `/pd_fsl/left_hand/{index}` -> `[x, y, z, visibility]`
- `/pd_fsl/right_hand/{index}` -> `[x, y, z, visibility]`
- `/pd_fsl/meta/person_count` -> integer
- `/pd_fsl/meta/timestamp` -> unix time float

Use these addresses in Warudo's OSC receiver/mapping layer.
