import cv2
import numpy as np

CAMERA_INDEX = 0
MAX_MISSING = 15
MAX_DISTANCE = 120
SMOOTHING = 0.35
GREEN = (0, 255, 0)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = np.array(box, dtype=float)
        self.missing = 0

    @property
    def center(self):
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])

    def update(self, box):
        self.box = SMOOTHING * np.array(box, dtype=float) + (1 - SMOOTHING) * self.box
        self.missing = 0


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cascade.detectMultiScale(gray, 1.2, 6, minSize=(70, 70))


def match(tracks, detections):
    unmatched = list(range(len(detections)))
    for track in tracks:
        if not unmatched:
            track.missing += 1
            continue
        dists = [
            np.linalg.norm(track.center - (detections[i][:2] + detections[i][2:] / 2))
            for i in unmatched
        ]
        best = int(np.argmin(dists))
        if dists[best] < MAX_DISTANCE:
            track.update(detections[unmatched.pop(best)])
        else:
            track.missing += 1
    return unmatched


def draw(frame, track):
    x, y, w, h = track.box.astype(int)
    L = max(12, min(w, h) // 5)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 1)
    for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), GREEN, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), GREEN, 3)
    mx, my = x + w // 2, y + h // 2
    cv2.line(frame, (mx - 8, my), (mx + 8, my), GREEN, 1)
    cv2.line(frame, (mx, my - 8), (mx, my + 8), GREEN, 1)
    cv2.putText(frame, f"TARGET {track.id:02d}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    tracks = []
    next_id = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = np.array(detect(frame)).reshape(-1, 4)
        unmatched = match(tracks, detections)

        for i in unmatched:
            tracks.append(Track(next_id, detections[i]))
            next_id += 1

        tracks = [t for t in tracks if t.missing <= MAX_MISSING]

        for track in tracks:
            if track.missing == 0:
                draw(frame, track)

        cv2.putText(frame, f"Faces: {sum(t.missing == 0 for t in tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.imshow("Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
