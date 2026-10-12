import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

FACE_IDS = {
    "nose": 1, "chin": 152, "forehead": 10,
    "l_cheek": 234, "r_cheek": 454,
    "l_eye": 33, "r_eye": 263,
    "mouth_l": 61, "mouth_r": 291, "mouth_c": 13,
}
FINGERTIPS = [4, 8, 12, 16, 20]
N_DIST = len(FINGERTIPS) * len(FACE_IDS)
HAND_DIM = 21 * 3 + N_DIST + 1
# Upper-body pose landmarks. Pose still estimates wrists/elbows when the hand
# models can't see the hands (e.g. hands behind head).
POSE_IDS = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
}
POSE_DIM = len(POSE_IDS) * 3 + 1          # (x, y, visibility) per point + present flag
FEATURE_DIM = len(FACE_IDS) * 3 + 1 + 2 * HAND_DIM + POSE_DIM   # 278


def _face_frame(face_lms):
    pts = np.array([[l.x, l.y, l.z] for l in face_lms.landmark])
    origin = pts[FACE_IDS["nose"]]
    scale = np.linalg.norm(pts[FACE_IDS["l_cheek"]] - pts[FACE_IDS["r_cheek"]]) + 1e-6
    le, re = pts[FACE_IDS["l_eye"]], pts[FACE_IDS["r_eye"]]
    roll = np.arctan2(re[1] - le[1], re[0] - le[0])
    return pts, origin, scale, roll


def _hand_features(hand_lms, origin, scale, face_pts):
    if hand_lms is None:
        return np.concatenate([np.zeros(63), np.full(N_DIST, 5.0), [0.0]])
    h = (np.array([[l.x, l.y, l.z] for l in hand_lms.landmark]) - origin) / scale
    key = (face_pts[list(FACE_IDS.values())] - origin) / scale
    d = np.linalg.norm(h[FINGERTIPS][:, None, :2] - key[None, :, :2], axis=2).flatten()
    return np.concatenate([h.flatten(), d, [1.0]])


def _pose_features(pose_lms, origin, scale):
    """Shoulders/elbows/wrists as xy in the face frame, plus visibility.
    z is skipped: pose z uses a different (hip-centred) depth scale than face z."""
    if pose_lms is None:
        return np.zeros(POSE_DIM)
    lms = [pose_lms.landmark[i] for i in POSE_IDS.values()]
    xy = (np.array([[l.x, l.y] for l in lms]) - origin[:2]) / scale
    vis = np.array([[l.visibility] for l in lms])
    return np.concatenate([np.hstack([xy, vis]).flatten(), [1.0]])


def extract_features(results):
    """Feature vector of length FEATURE_DIM, or None if no face detected."""
    if results.face_landmarks is None:
        return None
    face_pts, origin, scale, roll = _face_frame(results.face_landmarks)
    face_feats = ((face_pts[list(FACE_IDS.values())] - origin) / scale).flatten()
    left = _hand_features(results.left_hand_landmarks, origin, scale, face_pts)
    right = _hand_features(results.right_hand_landmarks, origin, scale, face_pts)
    pose = _pose_features(results.pose_landmarks, origin, scale)
    return np.concatenate([face_feats, [roll], left, right, pose]).astype(np.float32)
