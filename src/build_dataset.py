import cv2, numpy as np
from tqdm import tqdm
from src.config import RAW_DIR, PROC_DIR
from src.features import mp_holistic, extract_features


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    X, y = [], []
    paths = [(p, d.name) for d in sorted(RAW_DIR.iterdir()) if d.is_dir()
             for p in d.glob("*.[jp][pn]g")]
    skipped = 0
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holo:
        for path, label in tqdm(paths):
            img = cv2.imread(str(path))
            if img is None:
                continue
            f = extract_features(holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if f is None:
                skipped += 1
                continue
            X.append(f); y.append(label)
    np.save(PROC_DIR / "X.npy", np.array(X))
    np.save(PROC_DIR / "y.npy", np.array(y))
    print(f"Saved {len(X)} samples ({skipped} skipped, no face found)")


if __name__ == "__main__":
    main()
