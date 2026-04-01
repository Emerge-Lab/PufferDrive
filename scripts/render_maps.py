"""Render all maps showing road discretization: lanes, edges, and grid cells."""
import os
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pufferlib.ocean.drive import drive

map_dir = "pufferlib/resources/drive/binaries/carla"
maps = sorted(f for f in os.listdir(map_dir) if f.endswith(".bin"))

for map_file in maps:
    map_path = os.path.join(map_dir, map_file)
    map_name = map_file.replace(".bin", "")
    print(f"Rendering {map_name}...")

    tmpdir = tempfile.mkdtemp()
    shutil.copy(map_path, os.path.join(tmpdir, map_file))

    try:
        env = drive.Drive(
            num_agents=32,
            num_maps=1,
            map_dir=tmpdir,
            dynamics_model="jerk",
            simulation_mode="gigaflow",
            min_agents_per_env=1,
            max_agents_per_env=32,
            scenario_length=1024,
        )
        env.reset()
        state = env.get_state()
        if isinstance(state, list):
            state = state[0]

        road_elements = state.get("road_elements", [])
        type_counts = Counter()
        for elem in road_elements:
            type_counts[elem.get("type", -1)] += 1
        print(f"  Road element types: {dict(type_counts)}")

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set_facecolor('#1a1a1a')
        ax.set_aspect('equal')
        ax.set_title(f"{map_name} — Road Discretization", fontsize=16, color='white')

        # Color by type
        type_colors = {
            # Lanes (drivable)
            1: ('green', 'Drivable Lane', 0.6),
            2: ('limegreen', 'Lane Type 2', 0.6),
            3: ('darkgreen', 'Lane Type 3', 0.6),
            # Lines
            11: ('yellow', 'Road Line', 0.4),
            12: ('gold', 'Line Type 12', 0.4),
            13: ('orange', 'Line Type 13', 0.4),
            14: ('darkorange', 'Line Type 14', 0.4),
            15: ('khaki', 'Line Type 15', 0.4),
            16: ('wheat', 'Line Type 16', 0.4),
            17: ('lightyellow', 'Line Type 17', 0.4),
            18: ('palegoldenrod', 'Line Type 18', 0.4),
            # Edges
            21: ('red', 'Road Edge', 0.8),
            22: ('orangered', 'Edge Type 22', 0.8),
            23: ('darkred', 'Edge Type 23', 0.8),
        }

        plotted_types = set()
        for elem in road_elements:
            x = elem.get("x", [])
            y = elem.get("y", [])
            t = elem.get("type", 0)
            if not x or not y:
                continue

            color, label, lw = type_colors.get(t, ('gray', f'Type {t}', 0.3))
            show_label = t not in plotted_types
            plotted_types.add(t)
            ax.plot(x, y, color=color, linewidth=lw, alpha=0.7,
                    label=label if show_label else None)

        ax.legend(loc='upper right', fontsize=10, facecolor='#333', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        out_path = f"/tmp/map_{map_name}_discretized.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        print(f"  Saved to {out_path}")
        env.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {e}")
    finally:
        shutil.rmtree(tmpdir)

print("\nDone! Opening all maps...")
for map_file in maps:
    map_name = map_file.replace(".bin", "")
    os.system(f"open /tmp/map_{map_name}_discretized.png")
