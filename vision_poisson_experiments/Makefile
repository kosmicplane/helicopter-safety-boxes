PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: install verify-layout generate-demo test test-boxes test-project static-demo static-smoke live-demo-file clean

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e ../poisson_safety_box
	$(VENV_PIP) install -e ../cbf_safety_box
	$(VENV_PIP) install -e ".[dev]"
	@echo "Installed both sibling Safety Boxes and vision_poisson_experiments."


verify-layout:
	$(VENV_PYTHON) scripts/verify_workspace_layout.py

generate-demo:
	$(VENV_PYTHON) scripts/generate_synthetic_assets.py

test-boxes:
	cd ../poisson_safety_box && PYTHONPATH=. $(abspath $(VENV_PYTHON)) -m pytest -q
	cd ../cbf_safety_box && PYTHONPATH=. $(abspath $(VENV_PYTHON)) -m pytest -q

test-project:
	$(VENV_PYTHON) -m pytest -q

test: test-boxes test-project

static-demo: generate-demo
	rm -rf sample_outputs/static_demo
	$(VENV_PYTHON) 01_static_image_poisson_cbf/run_experiment.py \
		--image examples/assets/static_scene.png \
		--config 01_static_image_poisson_cbf/config_synthetic.yaml \
		--output sample_outputs/static_demo \
		--assume-top-down --headless


static-smoke: generate-demo
	rm -rf sample_outputs/static_smoke
	$(VENV_PYTHON) 01_static_image_poisson_cbf/run_experiment.py \
		--image examples/assets/static_scene.png \
		--config 01_static_image_poisson_cbf/config_smoke.yaml \
		--output sample_outputs/static_smoke \
		--assume-top-down --headless

live-demo-file: generate-demo
	rm -rf sample_outputs/live_demo
	$(VENV_PYTHON) 02_phone_stream_poisson_realtime/run_experiment.py \
		--source examples/assets/live_scene.avi \
		--config 02_phone_stream_poisson_realtime/config_synthetic.yaml \
		--output sample_outputs/live_demo \
		--headless --max-frames 60

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
