import os
from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("[0-9]*.py"))

# Kernel runs headless from notebooks/ (05 loads ../weights/...); repo root on
# PYTHONPATH so `from notebooks.notebook_utils import ...` resolves.
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")] if p)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(notebook_path):
    notebook = jupytext.read(notebook_path)
    NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    ).execute()
