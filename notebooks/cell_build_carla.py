# ── build_carla_figure: road map + N simulated GIGAFLOW agents ────────────────

AGENT_PALETTE = pc.qualitative.Plotly   # 10 distinct colours
GOAL_COLORS   = ['#FFD700', '#FFA500', '#FF6347']   # gold / orange / tomato


def build_carla_figure(scenario: dict, n_agents: int = 8, seed: int = 42,
                        mode: str = 'static',
                        show_arrows: bool = True, arrow_spacing: float = 6.0,
                        title: str = '') -> go.Figure:
    """
    Road map (build_figure) overlaid with n_agents simulated GIGAFLOW episodes.

    mode='static'  Straight route goals (default).
    mode='jump'    Route includes a lane-change injection (build_path_w_jump).
                   Pre-jump path is drawn solid; post-jump path is dotted.
                   The lane-change point is marked with a magenta triangle.

    Per agent (distinct colour):
      ─── Path from spawn  (solid pre-jump, dotted post-jump in jump mode)
      ▲   Lane-change point (jump mode only, magenta)
      ★   Spawn position
      ◆   Goal 1 / 2 / 3  (gold / orange / tomato)
      ---  Spawn → Goal 1 dashed line
    """
    fig = build_figure(scenario, show_arrows=show_arrows,
                       arrow_spacing=arrow_spacing, title=title)

    road_map = scenario['road_map']
    rng      = np.random.default_rng(seed)
    legend_seen = set()

    for agent_idx in range(n_agents):
        agent_color = AGENT_PALETTE[agent_idx % len(AGENT_PALETTE)]
        ag = simulate_gigaflow_agent(road_map, rng, mode=mode)
        if ag is None:
            print(f'  [warn] agent {agent_idx}: simulation failed, skipping')
            continue

        sx, sy   = ag['spawn_x'], ag['spawn_y']
        path_fwd = ag['path_fwd']
        goals    = ag['goals']
        spacings = ag['spacings']
        jump_fwd_idx = ag.get('jump_fwd_idx')  # None in static mode

        # ── Route path ────────────────────────────────────────────────────────
        if mode == 'jump' and jump_fwd_idx is not None and 1 < jump_fwd_idx < len(path_fwd):
            pre  = path_fwd[:jump_fwd_idx + 1]
            post = path_fwd[jump_fwd_idx:]

            pk = 'agent_path_pre'
            fig.add_trace(go.Scatter(
                x=pre[:, 0].tolist(), y=pre[:, 1].tolist(), mode='lines',
                line=dict(color=agent_color, width=2.5),
                name='Path (pre-jump)',
                legendgroup=pk, showlegend=(pk not in legend_seen),
                hovertext=f'Agent {agent_idx}  pre-jump  {len(pre)} wps',
                hoverinfo='text', opacity=0.9,
            ))
            legend_seen.add(pk)

            pk2 = 'agent_path_post'
            fig.add_trace(go.Scatter(
                x=post[:, 0].tolist(), y=post[:, 1].tolist(), mode='lines',
                line=dict(color=agent_color, width=2.5, dash='dot'),
                name='Path (post-jump)',
                legendgroup=pk2, showlegend=(pk2 not in legend_seen),
                hovertext=f'Agent {agent_idx}  post-jump  {len(post)} wps',
                hoverinfo='text', opacity=0.9,
            ))
            legend_seen.add(pk2)

            # Lane-change marker
            jx, jy = float(path_fwd[jump_fwd_idx, 0]), float(path_fwd[jump_fwd_idx, 1])
            jk = 'jump_point'
            fig.add_trace(go.Scatter(
                x=[jx], y=[jy], mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='#E040FB',
                            line=dict(color='white', width=1)),
                name='Lane-change point',
                legendgroup=jk, showlegend=(jk not in legend_seen),
                hovertext=f'Agent {agent_idx}  lane change @ fwd_idx={jump_fwd_idx}',
                hoverinfo='text',
            ))
            legend_seen.add(jk)

        else:
            # Static mode (or jump mode with no parallel lane found)
            pk = 'agent_path'
            fig.add_trace(go.Scatter(
                x=path_fwd[:, 0].tolist(), y=path_fwd[:, 1].tolist(),
                mode='lines',
                line=dict(color=agent_color, width=2.5),
                name='Spawned agent path',
                legendgroup=pk, showlegend=(pk not in legend_seen),
                hovertext=(f'Agent {agent_idx}  route={len(ag["route"])} lanes  '
                           f'path_len={path_fwd[-1,3]-path_fwd[0,3]:.0f} m'),
                hoverinfo='text', opacity=0.9,
            ))
            legend_seen.add(pk)

        # ── Spawn marker ──────────────────────────────────────────────────────
        sk = 'spawn_pos'
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy], mode='markers',
            marker=dict(symbol='star', size=14, color='#00E676',
                        line=dict(color='white', width=1)),
            name='Spawn position',
            legendgroup=sk, showlegend=(sk not in legend_seen),
            hovertext=f'Agent {agent_idx} spawn  lane={ag["spawn_lane"]}',
            hoverinfo='text',
        ))
        legend_seen.add(sk)

        # ── Goal markers + Spawn→Goal1 line ───────────────────────────────────
        for g_idx, (gx, gy) in enumerate(goals):
            gc    = GOAL_COLORS[g_idx % len(GOAL_COLORS)]
            gk    = f'goal_{g_idx}'
            dist_m = float(np.hypot(gx - sx, gy - sy))
            fig.add_trace(go.Scatter(
                x=[gx], y=[gy], mode='markers',
                marker=dict(symbol='diamond', size=12, color=gc,
                            line=dict(color='white', width=1)),
                name=f'Goal {g_idx + 1}',
                legendgroup=gk, showlegend=(gk not in legend_seen),
                hovertext=(f'Agent {agent_idx} goal {g_idx+1}  '
                           f'spacing={spacings[g_idx]:.1f} m  '
                           f'dist_from_spawn={dist_m:.1f} m'),
                hoverinfo='text',
            ))
            legend_seen.add(gk)

            if g_idx == 0:
                lk = 'spawn_goal_line'
                fig.add_trace(go.Scatter(
                    x=[sx, gx], y=[sy, gy], mode='lines',
                    line=dict(color=agent_color, width=1.0, dash='dash'),
                    name='Spawn → Goal 1',
                    legendgroup=lk, showlegend=(lk not in legend_seen),
                    hoverinfo='skip', opacity=0.5,
                ))
                legend_seen.add(lk)

    tid = scenario['metadata']['scenario_id']
    fig.update_layout(title=title or tid)
    return fig
