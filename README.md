# Seated Pose Classifier (face + hands)

MediaPipe Holistic extracts face, hand and upper-body (shoulder/elbow/wrist) landmarks; a PyTorch MLP (GPU if available) classifies the seated pose.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Pick the CUDA build matching `nvidia-smi` from https://pytorch.org
# RTX 50-series (Blackwell) needs cu128 or newer; older GPUs can use cu121
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```
Use Python 3.9-3.12.

## Workflow (run from project root)
```bash
python -m src.collect neutral          # SPACE or hold 'c' to save frames, 't' for hands-free timer, 'q' quits
python -m src.collect hand_on_chin     # ~150-300 images per pose
python -m src.build_dataset            # images -> data/processed/X.npy, y.npy
python -m src.train                    # saves models/pose_mlp.pt
python -m src.infer                    # live webcam, 'q' quits
```

`infer` shows an image for the detected pose beside the webcam feed. Put one image per label in
`assets/poses/`, named after the label (`neutral.png`, `hand_on_chin.jpg`, ...), plus optional
`unknown` and `no_face` images. Labels without an image show their name instead.

Add a new pose by running `collect` with a new label name; folders in `data/raw/` are the labels.
Don't mirror frames (`cv2.flip`) between training and inference.
Only rerun `build_dataset` when images or `features.py` change.
