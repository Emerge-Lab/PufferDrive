# Test entry points. Run `make help` (or bare `make`) for usage.
# Targets call .venv/bin/python directly — no venv activation needed.
PYTHON := .venv/bin/python

.DEFAULT_GOAL := help

# The compiled C extension the Python tests import. It depends on the sim
# sources, so `make test` rebuilds it only when a .c/.h file actually changed.
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
BINDING := pufferlib/ocean/drive/binding$(EXT_SUFFIX)
DRIVE_SOURCES := $(wildcard pufferlib/ocean/drive/*.c) $(wildcard pufferlib/ocean/drive/*.h) pufferlib/ocean/env_binding.h setup.py

.PHONY: help test rebuild test-unit test-eval test-c test-docker-smoke

help:
	@echo "PufferDrive test targets:"
	@echo ""
	@echo "  make test               Run the local suites (unit + eval + C), fail-fast."
	@echo "                          Rebuilds the C extension first if any .c/.h changed."
	@echo "  make rebuild            Force-rebuild the C extension unconditionally"
	@echo ""
	@echo "  make test-unit          Python unit tests (tests/unit_tests)"
	@echo "  make test-eval          Evaluation integration tests (tests/eval)"
	@echo "  make test-c             C sim tests (tests/drive: dynamics, geometry, IDM, ...)"
	@echo ""
	@echo "  make test-docker-smoke  Full smoke suite in Docker (train/rollout goldens)"

$(BINDING): $(DRIVE_SOURCES)
	$(PYTHON) setup.py build_ext --inplace --force

# Local suites, fast-failing first: Python unit tests, eval integration, C sim tests.
# The Docker smoke suite stays opt-in — it needs a Docker daemon.
test: test-unit test-eval test-c

rebuild:
	$(PYTHON) setup.py build_ext --inplace --force

test-unit: $(BINDING)
	$(PYTHON) -m pytest -v tests/unit_tests

test-eval: $(BINDING)
	$(PYTHON) -m pytest -v tests/eval

test-c:
	$(MAKE) -C tests/drive test

test-docker-smoke:
	@command -v docker >/dev/null || { \
		echo "error: docker not found. This suite runs in a Linux container —"; \
		echo "install Docker Desktop, or rely on the CI smoke job instead."; \
		exit 1; }
	docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
	docker run --rm pufferdrive-smoke
