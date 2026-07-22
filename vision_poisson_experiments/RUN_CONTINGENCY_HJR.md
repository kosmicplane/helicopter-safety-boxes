# Setup and Run Commands

Assume the sibling layout:

```text
Docker/Workspace/
├── poisson_safety_box/
├── cbf_safety_box/
└── vision_poisson_experiments/
```

## Installation

```bash
cd Docker/Workspace
python3 -m venv .venv_boxes
source .venv_boxes/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./poisson_safety_box
python -m pip install -e ./cbf_safety_box
python -m pip install -e './vision_poisson_experiments[dev]'
```

## Interactive phone / webcam / stream

```bash
cd vision_poisson_experiments
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source 'http://PHONE_IP:PORT/stream.mjpg' \
  --config 02_phone_stream_poisson_realtime/config.yaml
```

Webcam:

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source 0 \
  --config 02_phone_stream_poisson_realtime/config.yaml
```

Local video:

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_contingency_synthetic.yaml \
  --output sample_outputs/live_contingency_demo \
  --headless --max-frames 60
```

HTTP/MJPEG and RTSP sources continue to use the same `--source` argument.

## Deterministic five-scenario validation

```bash
python scripts/run_contingency_synthetic_validation.py \
  --output reports/contingency_validation
```

## Tests

```bash
python -m pytest -q
(cd ../cbf_safety_box && python -m pytest -q)
(cd ../poisson_safety_box && python -m pytest -q)
```

## Disable reachability

Set:

```yaml
reachability:
  enabled: false
```

The entrypoint then uses the original live Poisson pipeline.
