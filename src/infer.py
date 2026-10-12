import collections, cv2, numpy as np, torch
from src import config as C
from src.features import mp_holistic, extract_features, FEATURE_DIM
from src.model import PoseMLP

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
PANEL_BG = (30, 30, 30)


def _fit(img, w, h):
    """Resize img to fit inside w x h (keeping aspect ratio), centred on PANEL_BG.
    Transparent PNG pixels show the background."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas = np.full((h, w, 3), PANEL_BG, np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    roi = canvas[y0:y0 + nh, x0:x0 + nw]
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255
        roi[:] = (img[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
    else:
        roi[:] = img
    return canvas


def load_pose_images(labels, w, h):
    """{label: panel-sized image} for each label that has a file in POSE_IMAGE_DIR."""
    images = {}
    for label in labels:
        path = next((C.POSE_IMAGE_DIR / f"{label}{e}" for e in IMG_EXTS
                     if (C.POSE_IMAGE_DIR / f"{label}{e}").exists()), None)
        if path is None:
            continue
        # imdecode instead of imread so non-ASCII Windows paths work
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read {path}")
            continue
        images[label] = _fit(img, w, h)
    missing = [l for l in labels if l not in images]
    print(f"Pose images from {C.POSE_IMAGE_DIR}: {sorted(images) or 'none'}"
          + (f"; no image for {missing}" if missing else ""))
    return images


def render_panel(label, images, w, h):
    if label in images:
        return images[label]
    # No image for this label: show its name instead
    panel = np.full((h, w, 3), PANEL_BG, np.uint8)
    text = label.replace("_", " ")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(panel, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    return panel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.MODEL_PATH, map_location=device, weights_only=False)
    if ck["in_dim"] != FEATURE_DIM:
        raise SystemExit(f"Model expects {ck['in_dim']} features but features.py makes "
                         f"{FEATURE_DIM}. Rerun: python -m src.build_dataset; python -m src.train")
    model = PoseMLP(ck["in_dim"], len(ck["classes"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    classes, mean, scale = ck["classes"], ck["mean"], ck["scale"]

    recent = collections.deque(maxlen=C.SMOOTHING_WINDOW)
    cap = cv2.VideoCapture(0)
    images = None   # loaded on the first frame, once the frame height is known

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holo:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            h = frame.shape[0]
            if images is None:
                images = load_pose_images(list(classes) + ["unknown", "no_face"],
                                          C.PANEL_WIDTH, h)

            res = holo.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats = extract_features(res)

            label, text = "no_face", "no face"
            if feats is not None:
                x = torch.tensor(((feats - mean) / scale)[None], dtype=torch.float32,
                                 device=device)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                label = classes[probs.argmax()] if probs.max() > C.CONF_THRESHOLD else "unknown"
                recent.append(label)
                label = max(set(recent), key=recent.count)
                text = f"{label} ({probs.max():.0%})"

            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            panel = render_panel(label, images, C.PANEL_WIDTH, h)
            cv2.imshow("Seated pose", np.hstack([frame, panel]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
