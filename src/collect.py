"""Usage: python -m src.collect hand_on_chin
SPACE saves a frame, hold 'c' for continuous capture, 'q' quits.
't' starts a hands-free timer: countdown, then auto-capture for a few seconds
(press 't' again to cancel). Timing is set in config.py."""
import sys, time, cv2
from src.config import RAW_DIR, TIMER_COUNTDOWN_S, TIMER_BURST_S, TIMER_INTERVAL_S

label = sys.argv[1]
out = RAW_DIR / label
out.mkdir(parents=True, exist_ok=True)
n = len(list(out.glob("*.jpg")))

timer_start = None   # time 't' was pressed, None when the timer is off
last_shot = 0.0

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    save = False

    # Timer phases: countdown -> burst (save every TIMER_INTERVAL_S) -> off
    status = None
    if timer_start is not None:
        elapsed = now - timer_start
        if elapsed < TIMER_COUNTDOWN_S:
            status = (f"Get ready: {TIMER_COUNTDOWN_S - int(elapsed)}", (0, 255, 255))
        elif elapsed < TIMER_COUNTDOWN_S + TIMER_BURST_S:
            left = TIMER_COUNTDOWN_S + TIMER_BURST_S - elapsed
            status = (f"REC {left:.1f}s", (0, 0, 255))
            if now - last_shot >= TIMER_INTERVAL_S:
                save = True
                last_shot = now
        else:
            timer_start = None

    view = frame.copy()
    cv2.putText(view, f"{label}: {n} saved", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if status:
        cv2.putText(view, status[0], (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, status[1], 3)
    cv2.imshow("collect", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        timer_start = None if timer_start is not None else now
    if key in (ord(" "), ord("c")):
        save = True

    if save:
        # Save the clean frame (no overlay text); ms timestamp keeps names unique
        cv2.imwrite(str(out / f"{int(now * 1000)}.jpg"), frame)
        n += 1
cap.release()
cv2.destroyAllWindows()
