.PHONY: install test verify smoke paper predefined sweeps static live clean

install:
	./install.sh

test:
	./run_checks.sh

verify:
	python scripts/verify_clf_runtime.py

smoke:
	./scripts/run_smoke_suite.sh

paper:
	./scripts/run_paper_figures.sh

predefined:
	python experiments/predefined_world/run.py --profile smoke --output outputs/runs/predefined_smoke

sweeps:
	python experiments/predefined_world/run_sweeps.py --profile smoke --output outputs/runs/parameter_sweeps

static:
	python experiments/static_image/run.py --profile smoke --image experiments/static_image/input/example_scene.png --output outputs/runs/static_image

live:
	python experiments/live_vision/run.py --profile smoke --source experiments/live_vision/assets/example_stream.avi --output outputs/runs/live_vision

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
