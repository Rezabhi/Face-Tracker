# Face Tracker

Real-time face detection and tracking from your webcam, with a CCTV-style green lock-on box. Built with Python and OpenCV.

## Features

- Live face detection from the webcam
- Green corner-bracket box with a crosshair locked on each face
- Every face gets its own target ID that sticks as it moves
- Smooth box movement, no jitter
- Tracks multiple faces at once
- Runs on CPU, no GPU needed

## Requirements

- Python 3.9 or newer
- A webcam

## Installation

```bash
git clone https://github.com/<your-username>/face-tracker.git
cd face-tracker
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python face_tracker.py
```

Press `q` to quit.

## Configuration

These values sit at the top of `face_tracker.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | Which camera to use. Try `1` if the default does not open. |
| `MAX_MISSING` | `15` | Frames a target survives without being detected. |
| `MAX_DISTANCE` | `120` | Max pixel distance to match a detection to an existing target. |
| `SMOOTHING` | `0.35` | Box smoothing. Lower is smoother, higher is more responsive. |

## How it works

1. Each frame is converted to grayscale and equalized.
2. A Haar cascade classifier detects faces.
3. Detections are matched to existing targets by nearest center.
4. Matched boxes are smoothed, new faces get a new ID, and lost targets expire after a few frames.

## Notes

- Works best with good lighting and a face looking at the camera.
- This project only detects and follows faces. It does not identify who they are.

## License

MIT
