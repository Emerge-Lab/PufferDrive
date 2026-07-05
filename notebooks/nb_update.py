#!/usr/bin/env python3
"""Patch visualize_carla_geometry_goal.ipynb with jump-lane support."""
import nbformat

NB_PATH = '/scratch/yw4142/PufferDrive4/notebooks/visualize_carla_geometry_goal.ipynb'

with open('/tmp/cell_jump_helpers.py') as f:
    JUMP_SOURCE = f.read()
with open('/tmp/cell_build_carla.py') as f:
    BUILD_SOURCE = f.read()
with open('/tmp/cell_demo.py') as f:
    DEMO_SOURCE = f.read()

with open(NB_PATH) as f:
    nb = nbformat.read(f, as_version=4)

# Locate cells by unique marker strings
gigaflow_idx = build_idx = demo_idx = None
for i, cell in enumerate(nb.cells):
    src = cell.source
    if '_MIN_ROUTE_DIST = 60.0' in src and 'simulate_gigaflow_agent' in src:
        gigaflow_idx = i
    if 'def build_carla_figure' in src:
        build_idx = i
    if "TOWN = 'Town01'" in src and 'build_carla_figure(sc' in src:
        demo_idx = i

assert gigaflow_idx is not None, 'could not find GIGAFLOW cell'
assert build_idx     is not None, 'could not find build_carla_figure cell'
assert demo_idx      is not None, 'could not find demo cell'

print(f'Found cells: gigaflow={gigaflow_idx}  build={build_idx}  demo={demo_idx}')

# Insert new cell AFTER the gigaflow cell (highest index first to keep others stable)
# Update demo and build_carla_figure cells first (they come after gigaflow)
nb.cells[build_idx].source = BUILD_SOURCE
nb.cells[demo_idx].source  = DEMO_SOURCE

# Now insert the jump helpers cell right after the gigaflow cell
new_cell = nbformat.v4.new_code_cell(source=JUMP_SOURCE)
nb.cells.insert(gigaflow_idx + 1, new_cell)

with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)

print('Notebook patched successfully.')
print(f'  Total cells: {len(nb.cells)}')
for i, c in enumerate(nb.cells):
    snippet = c.source[:60].replace('\n', ' ')
    print(f'  [{i}] {snippet}...')
