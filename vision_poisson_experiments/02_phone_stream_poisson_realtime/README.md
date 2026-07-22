# Experiment 02 — Phone, Webcam, or Video Stream to Live Poisson Fields

This experiment accepts any source that `cv2.VideoCapture` can decode: webcam indices, local files, HTTP/MJPEG
streams, and RTSP streams. After calibration, the phone and workspace must remain fixed. An optional ORB/RANSAC
monitor invalidates the metric map when global camera motion exceeds configured thresholds.

The main thread performs capture, rectification, segmentation, temporal filtering, occupancy inflation, and display.
A single worker thread solves Poisson problems from a size-one latest-item queue. The UI never waits for a solve,
queued old maps are replaced, and completed obsolete solves are discarded.

## Phone or network stream

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source "http://PHONE_IP:PORT/video" \
  --config 02_phone_stream_poisson_realtime/config.yaml
```

## Headless file-based validation

```bash
python scripts/generate_synthetic_assets.py
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_synthetic.yaml \
  --output sample_outputs/live_demo \
  --headless --max-frames 60
```

Video FPS and Poisson update rate are reported separately. The optional `partial_h_t` calculation is a finite-difference
diagnostic only and is not inserted into the static CBF inequality. Low-rate 3D surfaces are produced only when a
snapshot is requested, keeping Matplotlib outside the per-frame display path.
