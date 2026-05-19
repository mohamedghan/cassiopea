# CASSI — Agents Reference

**C**amera-**A**ssisted **S**ervo **S**ystem **I**nterface.
A robotic hand control system that uses an Intel RealSense D435i depth camera and MediaPipe hand tracking to drive five servo motors on a prosthetic/robotic hand via an Arduino.

---

## Architecture Overview

```
RealSense D435i
  ├── Color stream (BGR, 640×480@30fps)  ──► MediaPipe HandLandmarker
  │                                              └── 21 landmarks (x, y, z_estimated)
  └── Depth stream (Z16, aligned to color) ──► depth injection
                                                 └── z_estimated → z_real (meters)
                                                       │
                                                       ▼
                                              Finger curl angles
                                                       │
                                         ┌─────────────┴──────────────┐
                                         ▼                            ▼
                                  One Euro Filter             Flask web UI
                                         │                    /video_feed
                                         ▼                    /status
                                  Arduino (serial)            /finger/<n>/<angle>
                                   A:t,i,m,r,p\n             /all/<angles>
                                         │
                                         ▼
                                  5× servo motors
                               (thumb, index, middle, ring, pinky)
```

---

## Repository Layout

```
CASSI/
├── src/hand_control/
│   ├── camera/
│   │   ├── realsense.py      # RealSenseCamera — pipeline, alignment, depth lookup
│   │   └── stream.py         # CameraStream — main loop, depth injection, MJPEG output
│   ├── tracking/
│   │   ├── tracker.py        # HandTracker — per-frame hand selection + One Euro Filter
│   │   ├── angles.py         # Finger curl → servo angle mapping (0–120)
│   │   └── visualization.py  # OpenCV landmark overlay, angle bars, distance HUD
│   ├── filters/
│   │   ├── one_euro.py       # One Euro Filter (adaptive, low-latency smoothing)
│   │   └── low_pass.py       # Simple low-pass filter (unused by default)
│   ├── arduino/
│   │   └── controller.py     # Serial protocol: "A:t,i,m,r,p\n" / "F:n,angle\n"
│   ├── web/
│   │   ├── routes.py         # Flask blueprint — video feed, status, manual override
│   │   └── templates/index.html
│   ├── models/
│   │   └── downloader.py     # Downloads hand_landmarker.task from Google storage
│   ├── app.py                # Flask application factory
│   ├── config.py             # Config dataclass — all values env-overridable
│   └── types.py              # LandmarkPoint protocol, MutableLandmark, FingerAngles
├── tests/
│   ├── conftest.py           # MockLandmark fixtures, hand pose fixtures
│   ├── test_angles.py        # Unit tests for angle calculations
│   └── test_filters.py       # Unit tests for One Euro Filter
├── ma.py                     # Standalone RealSense depth visualiser (dev tool)
├── Dockerfile                # Multi-stage build (python:3.12-slim)
├── docker-compose.yml        # Device passthrough, env vars, resource limits
├── pyproject.toml            # Dependencies, ruff, mypy, pytest config
└── requirements.txt          # Pinned versions (matches pyproject.toml deps)
```

---

## Key Modules

### `camera/realsense.py` — `RealSenseCamera`
Wraps the `pyrealsense2` pipeline.

- Enables **color** (`rs.format.bgr8`) and **depth** (`rs.format.z16`) streams at the configured resolution/fps.
- `rs.align(rs.stream.color)` aligns the depth frame to the color frame so every color pixel has a 1-to-1 depth value.
- `get_frames() -> (bgr_ndarray, rs.depth_frame)` — returns a synchronized pair; both are `None` on failure.
- `get_depth_at(depth_frame, px, py) -> float` — metric depth in meters; returns `0.0` for invalid/out-of-range pixels.

### `camera/stream.py` — `CameraStream` + `_inject_real_depth`
Main loop owner.

- Calls `realsense.get_frames()` each iteration for color + depth.
- Flips color horizontally (mirror effect) before feeding to MediaPipe. **The depth frame is never flipped.**
- `_inject_real_depth()` replaces every landmark's estimated `z` with the sensor reading.
  - **Coordinate un-flip**: landmark `x` is in flipped space; depth lookup uses `(1 - lm.x) * width` to map back to the unflipped depth frame.
  - Falls back to MediaPipe's estimated `z` only when the sensor returns `0` (occluded / out of range).
- Wrist landmark (`index 0`) depth is rendered on-screen as `Distance: X.XXX m`.

### `tracking/angles.py` — Angle calculation
- Landmark `z` values are now metric (meters) from the depth sensor.
- `calculate_angle(p1, p2, p3)` uses 3D dot-product; scale mismatch between normalized x/y and metric z is small in practice (adjacent joint depth differences are ~1–20 mm vs x/y deltas of ~50–150 px-normalised units).
- Servo mapping: `avg_joint_angle ∈ [60°, 180°] → servo ∈ [0, max_servo_angle]`.
- `max_servo_angle` defaults to `120` (config / `MAX_SERVO_ANGLE` env var).

### `tracking/tracker.py` — `HandTracker`
- Maintains a tracked palm center across frames using a weighted moving average.
- Selects the closest detected hand to the previous center; resets after 30 frames without detection.
- Applies an independent `OneEuroFilter` to each of the five finger angles.

### `arduino/controller.py` — `ArduinoController`
Serial protocol over `/dev/ttyACM0` at 115200 baud.

| Command | Format | Description |
|---------|--------|-------------|
| All angles | `A:t,i,m,r,p\n` | Send all five servo angles at once |
| Single finger | `F:n,angle\n` | Set one finger (0=thumb … 4=pinky) |

### `types.py`
- `LandmarkPoint` — Protocol satisfied by both MediaPipe landmarks and `MutableLandmark`.
- `MutableLandmark` — Dataclass (`x`, `y`, `z`) used after depth injection; `z` holds metric depth in meters.
- `FingerAngles` — `TypedDict` mapping finger name → servo angle (int, 0–120).

---

## Configuration

All fields in `config.py` are overridable via environment variable.

| Env var | Default | Description |
|---------|---------|-------------|
| `SERIAL_PORT` | `/dev/ttyACM0` | Arduino serial port |
| `BAUD_RATE` | `115200` | Serial baud rate |
| `REALSENSE_WIDTH` | `640` | Color + depth stream width |
| `REALSENSE_HEIGHT` | `480` | Color + depth stream height |
| `REALSENSE_FPS` | `30` | Stream frame rate |
| `MAX_SERVO_ANGLE` | `120` | Upper servo limit (maps to fully open) |
| `MODEL_PATH` | `hand_landmarker.task` | MediaPipe model file path |
| `HAND_DETECTION_CONFIDENCE` | `0.5` | MediaPipe detection threshold |
| `HAND_TRACKING_CONFIDENCE` | `0.5` | MediaPipe tracking threshold |
| `MAX_NUM_HANDS` | `2` | Max hands MediaPipe detects |
| `FILTER_MIN_CUTOFF` | `0.5` | One Euro Filter — smoothing at rest |
| `FILTER_BETA` | `0.01` | One Euro Filter — lag during fast motion |
| `FLASK_HOST` | `0.0.0.0` | Flask bind address |
| `FLASK_PORT` | `5000` | Flask port |
| `FLASK_DEBUG` | `false` | Flask debug mode |

---

## HTTP API

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Web UI |
| `/video_feed` | GET | MJPEG stream with tracking overlay |
| `/status` | GET | JSON — current angles + Arduino connection state |
| `/finger/<n>/<angle>` | GET | Set one finger (0–4) to angle (0–120) |
| `/all/<t,i,m,r,p>` | GET | Set all five angles in one request |

---

## Running Locally

**Prerequisites:**
- Python 3.12+
- Intel RealSense SDK udev rules installed on the host:
  ```bash
  sudo cp /path/to/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
- User in `plugdev` and `dialout` groups.

```bash
# Create and activate venv
python -m venv env && source env/bin/activate

# Install with dev extras
pip install -e ".[dev]"

# Download MediaPipe model (if not present)
python -c "from hand_control.models.downloader import download_model; download_model()"

# Run
python -m hand_control
```

Open `http://localhost:5000` in a browser.

---

## Running with Docker

```bash
docker compose up --build
```

The compose file passes `/dev/bus/usb` for RealSense access (pyrealsense2 uses libusb, not V4L2) and `/dev/ttyACM0` for the Arduino. The container joins the `plugdev` and `dialout` groups.

---

## Tests

```bash
pytest                  # run all tests
pytest -v tests/        # verbose
pytest --cov            # with coverage
```

Tests are pure-Python and do not require a physical camera or Arduino. `MockLandmark` in `conftest.py` satisfies the `LandmarkPoint` protocol and is also structurally identical to `MutableLandmark`.

---

## Adding Features — Conventions

- **New config values**: add a field to `Config` in `config.py` with a default, then add the corresponding `os.getenv(...)` line in `from_env()`.
- **New landmark data**: use `MutableLandmark` — it satisfies `LandmarkPoint` and works with all existing tracking and angle code without modification.
- **New serial commands**: extend `ArduinoController` in `arduino/controller.py`; update the Arduino firmware to match.
- **Depth access outside `stream.py`**: import `RealSenseCamera` from `hand_control.camera`; call `get_depth_at(depth_frame, px, py)` — always check the return value is `> 0` before trusting it.
- **Type checking**: `pyrealsense2` has no stubs; it is listed under `[[tool.mypy.overrides]]` with `ignore_missing_imports = true` in `pyproject.toml`.
