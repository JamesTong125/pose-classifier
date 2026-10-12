# Pose Classifier (face + hands)

A real-time computer vision app that recognizes seated poses (e.g. hand on chin, hand over mouth, face in hands) from a webcam feed, using only face and hand landmarks. Built in Python with MediaPipe Holistic and a PyTorch MLP which uses a GPU if it's available.

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
python -m src.collect neutral          # SPACE or hold 'c' to save frames, 't' for a hands-free timer, 'q' quits the program
python -m src.collect hand_on_chin     # Collect data for each pose, at least 100 images recommended
python -m src.build_dataset            # images get processed into a data set at data/processed/X.npy, y.npy
python -m src.train                    # saves the trained model to models/pose_mlp.pt
python -m src.infer                    # live webcam which detects your pose, 'q' quits the program
```

`infer` shows an image for the detected pose beside the webcam feed. Put an image per label in
`assets/poses/`, named after the label e.g: `neutral.png`, `hand_on_chin.jpg`, plus optional
`unknown` and `no_face` images. Labels without an image show their name instead.

Add a new pose by adding a folder in `data/raw/` and running `collect` with a new label name.
Don't mirror frames (`cv2.flip`) between training and inference.
Only rerun `build_dataset` when images are added or `features.py` change.
