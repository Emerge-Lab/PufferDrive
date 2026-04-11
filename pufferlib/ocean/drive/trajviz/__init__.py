"""trajviz — Vulkan offline renderer for saved Drive trajectories.

Public API:

    >>> from pufferlib.ocean.drive.trajviz import Renderer, render_npz
    >>> with Renderer(width=1280, height=720) as r:
    ...     r.render_episode(
    ...         road_xy=..., road_offsets=..., road_types=...,
    ...         traj_xyh=...,
    ...         agent_lengths=...,
    ...         out_topdown="td.mp4", out_bev="bev.mp4",
    ...     )

Or, more usually, the high-level npz path:

    >>> from pufferlib.ocean.drive.trajviz import render_npz
    >>> render_npz("trajectories_000010.npz",
    ...            maps_dir="path/to/maps",
    ...            out_dir="videos/")

CLI:

    python -m pufferlib.ocean.drive.trajviz <npz...> --maps-dir <dir> --out <dir>

The Vulkan context is created once per Renderer and reused across many
render_episode calls — pay the ~50 ms init cost once for a whole batch.

If the C extension fails to import (typically because libvulkan is not
installed at runtime, or the build did not include trajviz), this module
raises ImportError on first use with a pointer to docs/trajviz.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from pufferlib.ocean.drive import map_io


class _NativeUnavailable:
    """Stand-in raised when the C extension isn't built/available."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def __getattr__(self, name):
        raise ImportError(
            "trajviz._native is not available. Build with TRAJVIZ=1 and "
            "make sure libvulkan-dev + glslang-tools are installed. "
            f"Original error: {self._exc}\n"
            "See docs/trajviz.md for setup."
        )


try:
    from . import _native  # type: ignore
except ImportError as _e:
    _native = _NativeUnavailable(_e)  # type: ignore


class Renderer:
    """Vulkan trajectory renderer with a hot context across episodes.

    Use as a context manager to ensure the Vulkan context is closed even
    on exceptions:

        with Renderer(width=1280, height=720) as r:
            for episode in episodes:
                r.render_episode(...)
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self._ctx = _native.init(width, height)
        self.width = int(width)
        self.height = int(height)

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._ctx is not None:
            _native.close(self._ctx)
            self._ctx = None

    def render_batch(
        self,
        episodes: list,
        *,
        fps: int = 30,
    ) -> None:
        """Render N episodes simultaneously by tiling them in a per-view atlas.

        ``episodes`` is a list of dicts, each with these keys:

            road_xy:        (V, 2) float array (mean-centered sim frame)
            road_offsets:   (P+1,) int CSR offsets into road_xy
            road_types:     (P,)   int TVZ_ROAD_* type ids
            traj_xyh:       (T, A, 3) float (x, y, heading) per agent per step
            agent_lengths:  (A,)   int valid step counts (optional, defaults to T)
            ego_idx:        int (default -1 = first valid)
            out_topdown:    str path or None
            out_bev:        str path or None

        All episodes must use the same renderer dimensions (the Renderer's
        ``width`` × ``height``); inside, every episode's tile is exactly
        that size. Episodes with different ``T`` and ``A`` are accepted —
        the wrapper pads them to the batch's max T and A so the C extension
        can use a uniform shape.

        Calling this on the same Renderer with the same ``len(episodes)``
        across calls is much cheaper than recreating: the BatchRenderer's
        atlas + readback buffers are kept hot.
        """
        if self._ctx is None:
            raise RuntimeError("Renderer is closed")
        if not episodes:
            return
        batch_size = len(episodes)
        if batch_size > 16:
            raise ValueError(f"batch_size {batch_size} exceeds the v1 cap of 16")

        # Find the batch-wide max T and A so we can pad into a uniform tensor.
        num_steps = max(int(ep["traj_xyh"].shape[0]) for ep in episodes)
        max_agents = max(int(ep["traj_xyh"].shape[1]) for ep in episodes)

        traj = np.zeros((batch_size, num_steps, max_agents, 3), dtype=np.float32)
        agent_lengths = np.zeros((batch_size, max_agents), dtype=np.int32)
        for i, ep in enumerate(episodes):
            t = np.ascontiguousarray(ep["traj_xyh"], dtype=np.float32)
            T, A, _ = t.shape
            traj[i, :T, :A, :] = t
            if "agent_lengths" in ep and ep["agent_lengths"] is not None:
                ll = np.ascontiguousarray(ep["agent_lengths"], dtype=np.int32)
                agent_lengths[i, : len(ll)] = ll
            else:
                agent_lengths[i, :A] = T

        # Concatenate ragged road geometry with CSR-style per-episode offsets.
        # The C side splits each per-episode slice out of these flat arrays.
        all_xy_parts = []
        all_off_parts = []
        all_typ_parts = []
        vert_offsets = [0]
        poly_meta_offsets = [0]  # cumulative number of (num_polys+1) entries
        poly_type_offsets = [0]  # cumulative number of polys
        for ep in episodes:
            xy = np.ascontiguousarray(ep["road_xy"], dtype=np.float32)
            offs = np.ascontiguousarray(ep["road_offsets"], dtype=np.uint32)
            typs = np.ascontiguousarray(ep["road_types"], dtype=np.uint32)
            all_xy_parts.append(xy)
            all_off_parts.append(offs)
            all_typ_parts.append(typs)
            vert_offsets.append(vert_offsets[-1] + xy.shape[0])
            poly_meta_offsets.append(poly_meta_offsets[-1] + offs.shape[0])
            poly_type_offsets.append(poly_type_offsets[-1] + typs.shape[0])

        all_road_xy = np.concatenate(all_xy_parts, axis=0) if all_xy_parts else np.zeros((0, 2), np.float32)
        all_road_offsets = np.concatenate(all_off_parts) if all_off_parts else np.zeros((0,), np.uint32)
        all_road_types = np.concatenate(all_typ_parts) if all_typ_parts else np.zeros((0,), np.uint32)
        vert_offsets = np.array(vert_offsets, dtype=np.uint32)
        poly_meta_offsets = np.array(poly_meta_offsets, dtype=np.uint32)
        poly_type_offsets = np.array(poly_type_offsets, dtype=np.uint32)

        ego = np.array([int(ep.get("ego_idx", -1)) for ep in episodes], dtype=np.int32)
        out_td = [ep.get("out_topdown") for ep in episodes]
        out_bev = [ep.get("out_bev") for ep in episodes]

        _native.render_episodes_batch(
            self._ctx,
            all_road_xy=all_road_xy,
            vert_offsets=vert_offsets,
            all_road_offsets=all_road_offsets,
            poly_meta_offsets=poly_meta_offsets,
            all_road_types=all_road_types,
            poly_type_offsets=poly_type_offsets,
            traj_xyh=traj,
            agent_lengths=agent_lengths,
            ego_idx_per_ep=ego,
            fps=int(fps),
            out_topdown_paths=out_td,
            out_bev_paths=out_bev,
        )

    def render_episode(
        self,
        *,
        road_xy: np.ndarray,
        road_offsets: np.ndarray,
        road_types: np.ndarray,
        traj_xyh: np.ndarray,
        agent_dims: Optional[np.ndarray] = None,
        agent_lengths: Optional[np.ndarray] = None,
        ego_idx: int = -1,
        fps: int = 30,
        out_topdown: Optional[str] = None,
        out_bev: Optional[str] = None,
    ) -> None:
        """Render one episode to one or two MP4 files.

        Either ``out_topdown`` or ``out_bev`` (or both) must be set.
        """
        if self._ctx is None:
            raise RuntimeError("Renderer is closed")
        if out_topdown is None and out_bev is None:
            raise ValueError("must set at least one of out_topdown / out_bev")

        # The C extension is strict about dtypes / contiguity. Coerce here
        # so callers can pass float64 / non-contiguous slices without
        # tripping the validator.
        road_xy = np.ascontiguousarray(road_xy, dtype=np.float32)
        road_offsets = np.ascontiguousarray(road_offsets, dtype=np.uint32)
        road_types = np.ascontiguousarray(road_types, dtype=np.uint32)
        traj_xyh = np.ascontiguousarray(traj_xyh, dtype=np.float32)
        if agent_dims is not None:
            agent_dims = np.ascontiguousarray(agent_dims, dtype=np.float32)
        if agent_lengths is not None:
            agent_lengths = np.ascontiguousarray(agent_lengths, dtype=np.int32)

        _native.render_episode(
            self._ctx,
            road_xy=road_xy,
            road_offsets=road_offsets,
            road_types=road_types,
            traj_xyh=traj_xyh,
            agent_dims=agent_dims,
            agent_lengths=agent_lengths,
            ego_idx=int(ego_idx),
            fps=int(fps),
            out_topdown=out_topdown,
            out_bev=out_bev,
        )


# ---------------------------- npz convenience ---------------------------- #


def _resolve_map_path(name: str, maps_dir: Path) -> Optional[Path]:
    """Try a few likely locations for a map file referenced in the npz."""
    candidates = [Path(name), maps_dir / Path(name).name, maps_dir / name]
    for c in candidates:
        if c.exists():
            return c
    return None


def render_npz(
    npz_path: str | Path,
    maps_dir: str | Path,
    out_dir: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    views: Iterable[str] = ("topdown", "bev"),
    renderer: Optional[Renderer] = None,
) -> list[Path]:
    """Render every episode in a saved trajectories_*.npz file.

    Each env in the npz becomes one episode → one or two MP4 files in
    ``out_dir`` (named ``{npz_stem}_env{ID}_{view}.mp4``).

    If ``renderer`` is None, a fresh one is created and torn down inside
    the call. Pass an existing Renderer to amortize Vulkan startup over
    many .npz files.
    """
    npz_path = Path(npz_path)
    maps_dir = Path(maps_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = set(views)
    has_td = "topdown" in views
    has_bev = "bev" in views
    if not (has_td or has_bev):
        raise ValueError("views must include at least one of 'topdown', 'bev'")

    data = np.load(npz_path, allow_pickle=True)
    required = (
        "traj_x",
        "traj_y",
        "traj_heading",
        "traj_lengths",
        "agent_offsets",
        "map_ids",
        "map_files",
    )
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"{npz_path} missing keys {missing}. Has: {sorted(data.files)}")

    traj_x = data["traj_x"]
    traj_y = data["traj_y"]
    traj_heading = data["traj_heading"]
    traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int32)
    agent_offsets = np.asarray(data["agent_offsets"], dtype=np.int64)
    map_ids = np.asarray(data["map_ids"], dtype=np.int64)
    map_files = [str(m) for m in np.asarray(data["map_files"]).tolist()]

    num_envs = len(map_ids)
    # world_means (plural, per-env, shape (num_envs, 3)) is the right key —
    # each Drive sub-env in a vec has its own world_mean computed from its
    # own map's geometry, so different maps have different centerings. The
    # legacy single world_mean (env 0 only) leads to mis-aligned roads for
    # any env_id != 0 with a different map. Prefer the new key; fall back
    # to the legacy one with a warning so older saved npz files still
    # render (incorrectly for non-env-0, but at least they render).
    if "world_means" in data.files:
        world_means = np.asarray(data["world_means"], dtype=np.float32)
        if world_means.shape != (num_envs, 3):
            raise ValueError(f"world_means has shape {world_means.shape}, expected ({num_envs}, 3)")
    elif "world_mean" in data.files:
        legacy = np.asarray(data["world_mean"], dtype=np.float32)
        if num_envs > 1:
            print(
                f"  WARNING: {npz_path.name} has only the legacy single "
                f"world_mean key (env 0). Roads for non-env-0 trajectories "
                f"with different maps will be mis-aligned. Re-save with the "
                f"current pufferl to get per-env world_means."
            )
        world_means = np.broadcast_to(legacy[None, :], (num_envs, 3)).copy()
    else:
        raise ValueError(f"{npz_path} has neither world_means nor world_mean")

    if len(agent_offsets) == num_envs:
        # Some saved npz omit the trailing offset; reconstruct from the
        # total agent count.
        agent_offsets = np.concatenate([agent_offsets, [traj_x.shape[0]]])
    elif len(agent_offsets) != num_envs + 1:
        raise ValueError(f"agent_offsets length {len(agent_offsets)} doesn't match num_envs {num_envs}")

    own_renderer = renderer is None
    if own_renderer:
        renderer = Renderer(width=width, height=height)

    out_paths: list[Path] = []
    # Cache key includes the per-env world_mean tuple, not just the
    # map_id, so that if two sub-envs ever shared a map_id but used
    # different world_means (edge case under heterogeneous init_modes)
    # we don't return the wrong centering.
    map_cache: dict[tuple, tuple] = {}

    try:
        for env_id in range(num_envs):
            a0, a1 = int(agent_offsets[env_id]), int(agent_offsets[env_id + 1])
            if a1 <= a0:
                continue

            mid = int(map_ids[env_id])
            wm_env = world_means[env_id]
            cache_key = (mid, float(wm_env[0]), float(wm_env[1]), float(wm_env[2]))
            if cache_key not in map_cache:
                mp = _resolve_map_path(map_files[mid], maps_dir)
                if mp is None:
                    print(f"  env {env_id}: map {map_files[mid]} not found, skipping")
                    continue
                roads_raw = map_io.load_map_roads(mp)
                roads = map_io.mean_center_roads(roads_raw, wm_env)
                map_cache[cache_key] = map_io.roads_to_csr(roads)
            road_xy, road_offsets, road_types = map_cache[cache_key]

            # Slice trajectories for this env and stack into (T, A, 3)
            ex = traj_x[a0:a1]
            ey = traj_y[a0:a1]
            eh = traj_heading[a0:a1]
            elen = traj_lengths[a0:a1]
            num_agents, num_steps = ex.shape
            traj_xyh = np.empty((num_steps, num_agents, 3), dtype=np.float32)
            traj_xyh[..., 0] = ex.T
            traj_xyh[..., 1] = ey.T
            traj_xyh[..., 2] = eh.T

            scenario = f"{npz_path.stem}_env{env_id:03d}"
            td_path = (out_dir / f"{scenario}_topdown.mp4") if has_td else None
            bev_path = (out_dir / f"{scenario}_bev.mp4") if has_bev else None

            renderer.render_episode(
                road_xy=road_xy,
                road_offsets=road_offsets,
                road_types=road_types,
                traj_xyh=traj_xyh,
                agent_lengths=elen,
                ego_idx=-1,
                fps=fps,
                out_topdown=str(td_path) if td_path else None,
                out_bev=str(bev_path) if bev_path else None,
            )

            if td_path:
                out_paths.append(td_path)
            if bev_path:
                out_paths.append(bev_path)
            print(f"  env {env_id}: {num_agents} agents, {num_steps} steps")

    finally:
        if own_renderer:
            renderer.close()

    return out_paths


__all__ = ["Renderer", "render_npz"]
