# ── Visualise one map — switch between 'static' and 'jump' goal modes ─────────
# GOAL_MODE='jump'  works best on towns with multi-lane roads.
# Hit rates: Town01 ~5%  Town03 ~80%  Town05 ~95%
TOWN      = 'Town03'
GOAL_MODE = 'jump'   # 'static' | 'jump'
sc        = carla_maps[TOWN]

fig = build_carla_figure(sc, n_agents=10, seed=42, mode=GOAL_MODE,
                          arrow_spacing=6.0,
                          title=f'{TOWN} — GIGAFLOW {GOAL_MODE} mode')
fig.show()

# ── Side-by-side distance summary ─────────────────────────────────────────────
for mode_label in ('static', 'jump'):
    rng = np.random.default_rng(42)
    print(f'\n=== {TOWN}: {mode_label} mode  (10 agents) ===')
    for i in range(10):
        ag = simulate_gigaflow_agent(sc['road_map'], rng, mode=mode_label)
        if ag is None:
            print(f'  Agent {i:2d}  FAILED')
            continue
        sx, sy = ag['spawn_x'], ag['spawn_y']
        dists  = '  '.join(
            f'g{j+1}={np.hypot(float(g[0])-sx, float(g[1])-sy):.0f} m'
            for j, g in enumerate(ag['goals']))
        jinfo = (f"  jump@fwd={ag['jump_fwd_idx']}" if mode_label == 'jump' else '')
        print(f'  Agent {i:2d}  {dists}{jinfo}')
