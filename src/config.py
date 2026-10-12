from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "pose_mlp.pt"
# infer.py side panel: put <label>.png/.jpg here (also "unknown" and "no_face")
POSE_IMAGE_DIR = ROOT / "assets" / "poses"
PANEL_WIDTH = 480

EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
CONF_THRESHOLD = 0.6
SMOOTHING_WINDOW = 7

# collect.py hands-free timer ('t'): countdown, then auto-capture burst
TIMER_COUNTDOWN_S = 3
TIMER_BURST_S = 5
TIMER_INTERVAL_S = 0.1    # ~10 frames/s during the burst
