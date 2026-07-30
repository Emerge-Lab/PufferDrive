# Test entry points. Run `make help` (or bare `make`) for usage.
# Targets call .venv/bin/python directly — no venv activation needed.
PYTHON := .venv/bin/python

.DEFAULT_GOAL := help

# The compiled C extension the Python tests import. It depends on the sim
# sources, so `make test` rebuilds it only when a .c/.h file actually changed.
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
BINDING := pufferlib/ocean/drive/binding$(EXT_SUFFIX)
DRIVE_SOURCES := $(wildcard pufferlib/ocean/drive/*.c) $(wildcard pufferlib/ocean/drive/*.h) setup.py

.PHONY: help test rebuild ensure-test-deps test-unit test-c test-smoke test-notebooks test-docker-smoke

help:
	@echo "PufferDrive test targets:"
	@echo ""
	@echo "  make test               Run the local suites (unit + C + smoke + notebooks),"
	@echo "                          fail-fast. Rebuilds the C extension first if any"
	@echo "                          .c/.h changed; installs test deps if missing."
	@echo "  make rebuild            Force-rebuild the C extension unconditionally"
	@echo ""
	@echo "  make test-unit          Python unit tests (tests/unit_tests)"
	@echo "  make test-c             C sim tests (tests/drive: dynamics, geometry, IDM, ...)"
	@echo "  make test-smoke         Replay-HTML smoke test"
	@echo "  make test-notebooks     Execute every checked-in notebook end-to-end"
	@echo ""
	@echo "  make test-docker-smoke  Full smoke suite in Docker (train/rollout/eval goldens)"

$(BINDING): $(DRIVE_SOURCES)
	$(PYTHON) setup.py build_ext --inplace --force

# Local suites, fast-failing first: Python unit tests, C sim tests,
# replay-HTML smoke test, then the notebook executions (slowest). The
# Docker smoke suite stays opt-in — it needs a Docker daemon.
test: test-unit test-c test-smoke test-notebooks

rebuild:
	$(PYTHON) setup.py build_ext --inplace --force

# Notebook deps (the `test` extra in pyproject.toml) are installed on demand
# so `make test` works from a bare `uv pip install -e .` checkout.
ensure-test-deps:
	@$(PYTHON) -c "import jupytext, nbclient, ipykernel" 2>/dev/null || uv pip install -e '.[test]'

test-unit: $(BINDING)
	$(PYTHON) -m pytest -v tests/unit_tests

test-c:
	$(MAKE) -C tests/drive test

test-smoke: $(BINDING)
	$(PYTHON) -m pytest -v tests/smoke_tests/test_validation_replay_html.py

test-notebooks: $(BINDING) ensure-test-deps
	$(PYTHON) -m pytest -v tests/notebooks

test-docker-smoke:
	@command -v docker >/dev/null || { \
		echo "error: docker not found. This suite runs in a Linux container —"; \
		echo "install Docker Desktop, or rely on the CI smoke job instead."; \
		exit 1; }
	docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
	docker run --rm pufferdrive-smoke
