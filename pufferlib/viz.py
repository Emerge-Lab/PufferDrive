import dataclasses
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


ENTITY_COLORS = {
    1: np.array([65, 105, 225]) / 255.0,  # Vehicle (Royal Blue)
    2: np.array([0, 255, 0]) / 255.0,  # Pedestrian (Green)
    3: np.array([255, 0, 255]) / 255.0,  # Cyclist (Magenta)
    4: np.array([211, 211, 211]) / 255.0,  # Road lane (Light Gray)
    5: np.array([80, 80, 80]) / 255.0,  # Road Line (Gray)
    6: np.array([0, 0, 0]) / 255.0,  # Road Edge (Black)
    7: np.array([255, 0, 0]) / 255.0,  # Stop Sign (Red)
}

AGENT_COLOR = {
    "ok": np.array([65, 105, 225]) / 255.0,  # Vehicle (Royal Blue)
    "collision": np.array([255, 0, 0]) / 255.0,  # Vehicle in collision (Red)
    "inactive": np.array([211, 211, 211]) / 255.0,  # Inactive vehicle (Light Gray)
    "static": np.array([128, 128, 128]) / 255.0,  # Static vehicle (Gray)
}


@dataclasses.dataclass
class VizConfig:
    """Config for visualization."""

    front_x: float = 75.0
    back_x: float = 75.0
    front_y: float = 75.0
    back_y: float = 75.0
    px_per_meter: float = 10.0
    show_agent_id: bool = True
    center_agent_idx: int = -1


def init_fig_ax_via_size(x_px: float, y_px: float) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Initializes a figure with given size in pixel."""
    fig, ax = plt.subplots()
    # Sets output image to pixel resolution.
    dpi = 200
    fig.set_size_inches([x_px / dpi, y_px / dpi])
    fig.set_dpi(dpi)
    fig.set_facecolor("white")

    return fig, ax


def init_fig_ax(vis_config: VizConfig = VizConfig()) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Initializes a figure with vis_config."""
    return init_fig_ax_via_size(
        (vis_config.front_x + vis_config.back_x) * vis_config.px_per_meter,
        (vis_config.front_y + vis_config.back_y) * vis_config.px_per_meter,
    )


def img_from_fig(fig) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_argb()."""
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    plt.close(fig)

    return img


def plot_bounding_boxes(
    ax,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha: float = 1.0,
    zorder: int = 3,
    label=None,
) -> None:
    """Plots multiple bounding boxes.

    Args:
      ax: Fig handles.
      bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
        yaw).
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
      label: String, represents the meaning of the color for different boxes.
    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5:
        raise ValueError(
            ("Expect bboxes rank 2, last dimension of bbox 5 got{}, {}, {} respectively").format(
                bboxes.ndim, bboxes.shape[1], color.shape
            )
        )

    c = np.cos(bboxes[:, 4])
    s = np.sin(bboxes[:, 4])
    pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = np.array((c, s))
    ut = np.array((s, -c))

    # Compute box corner coordinates.
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Compute heading arrow using center left/right/front.
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u

    # Draw bboxes.
    ax.plot(
        [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
        [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
        color=color,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )

    # Draw heading arrow.
    ax.plot(
        [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
        [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
        color=color,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )


def plot_entity(ax, entity, idx, active_agent_indices, static_car_indices):
    entity_type = entity["type"]
    obj_color = ENTITY_COLORS.get(entity_type, "pink")

    # Vehicle
    if entity_type == 1:
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])
        length = np.array(entity["length"])
        width = np.array(entity["width"])
        heading = np.array(entity["heading"])
        goal_x = np.array(entity["goal_position_x"])
        goal_y = np.array(entity["goal_position_y"])

        if idx in active_agent_indices:
            obj_color = AGENT_COLOR["ok"]
            ax.scatter(goal_x, goal_y, s=20, color=obj_color, marker="o")
            ax.add_patch(Circle((goal_x, goal_y), radius=2.0, color=obj_color, fill=False, linestyle="--"))
        elif idx in static_car_indices:
            obj_color = AGENT_COLOR["static"]
        else:
            obj_color = AGENT_COLOR["inactive"]

        if entity["collision_state"]:
            obj_color = AGENT_COLOR["collision"]

        bbox = np.array((x, y, length, width, heading)).reshape(1, 5)
        plot_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5, zorder=4)

    # Pedestrian
    if entity_type == 2:
        return  # TODO Enable pedestrian visualization
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])

        ax.scatter(x, y, color=obj_color, s=20, marker="o", zorder=3)

    # Cyclist
    if entity_type == 3:
        return  # TODO Enable cyclist visualization
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])

        ax.scatter(x, y, color=obj_color, s=200, marker="o", zorder=3)
        # bbox = np.array((x, y, length, width, heading)).reshape(1, 5)
        # plot_numpy_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5)

    # Road lane
    if entity_type == 4:
        ax.plot(entity["traj_x"], entity["traj_y"], color=obj_color, linewidth=1, zorder=1)

    # Road line
    if entity_type == 5:
        ax.plot(entity["traj_x"], entity["traj_y"], color=obj_color, linewidth=1, linestyle="--", zorder=1)

    # Road edge
    if entity_type == 6:
        ax.plot(entity["traj_x"], entity["traj_y"], color=obj_color, linewidth=2, zorder=1)

    # Stop sign
    if entity_type == 7:
        ax.scatter(entity["traj_x"], entity["traj_y"], color=obj_color, s=150, marker="H", zorder=2)

    # Crosswalk
    if entity_type == 8:
        points = np.vstack((entity["traj_x"], entity["traj_y"])).T
        ax.add_patch(
            Polygon(
                xy=points,
                facecolor="none",
                edgecolor="xkcd:bluish grey",
                linewidth=2,
                alpha=0.4,
                hatch=r"//",
                zorder=2,
            )
        )

    # Speed bump
    if entity_type == 9:
        points = np.vstack((entity["traj_x"], entity["traj_y"])).T
        ax.add_patch(
            Polygon(
                xy=points,
                facecolor="xkcd:goldenrod",
                edgecolor="xkcd:black",
                linewidth=0,
                alpha=0.5,
                hatch=r"//",
                zorder=2,
            )
        )


def plot_simulator_state(scenario, viz_config: dict | None = None) -> np.ndarray:
    viz_config = VizConfig() if viz_config is None else VizConfig(**viz_config)
    # fig, ax = init_fig_ax(viz_config)

    fig, ax = plt.subplots(figsize=(20, 20))

    for idx, entity in enumerate(scenario["entities"]):
        plot_entity(ax, entity, idx, scenario["active_agent_indices"], scenario["static_car_indices"])

    ax.axis((-viz_config.back_x, viz_config.front_x, -viz_config.back_y, viz_config.front_y))
    ax.set_aspect("equal", adjustable="box")

    return img_from_fig(fig)


def unpack_obs(obs_flat):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
    Return:
        ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
    """
    ego_state = obs_flat[:, :7]

    max_cars = 63
    default_feature_size = 7
    max_road_segments = 200

    size_partners_obs = max_cars * default_feature_size
    partners_obs = obs_flat[:, 7 : 7 + size_partners_obs]
    partners_obs = partners_obs.reshape(-1, max_cars, default_feature_size)

    road_obs = obs_flat[
        :,
        7 + size_partners_obs : 7 + size_partners_obs + max_road_segments * default_feature_size,
    ]
    road_obs = road_obs.reshape(-1, max_road_segments, default_feature_size)

    return ego_state[0], partners_obs[0], road_obs[0]


def plot_observation(obs) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(20, 20))

    obs = unpack_obs(obs)

    ego_state, partners_obs, road_obs = obs

    # Plot ego
    goal_x, goal_y, ego_speed, ego_width, ego_length, _, _ = ego_state

    ego_heading = np.arctan2(0, 1)
    bbox = np.array((0, 0, ego_length, ego_width, ego_heading)).reshape(1, 5)
    obj_color = np.array([0, 0, 1])
    plot_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5)
    ax.scatter(goal_x, goal_y, color="red", marker="*", s=100)

    for i in range(partners_obs.shape[0]):
        if np.all(partners_obs[i] == 0):
            continue
        (
            partners_x,
            partners_y,
            partners_width,
            partners_length,
            partners_heading_x,
            partners_heading_y,
            partners_speed,
        ) = partners_obs[i]
        heading = np.arctan2(partners_heading_y, partners_heading_x)
        bbox = np.array((partners_x, partners_y, partners_length, partners_width, heading)).reshape(1, 5)
        plot_bounding_boxes(ax, bbox, color=np.array([0.5, 0.5, 0.5]), alpha=0.5)

    for i in range(road_obs.shape[0]):
        if np.all(road_obs[i] == 0):
            continue
        (
            road_x,
            road_y,
            road_length,
            road_width,
            road_heading_x,
            road_heading_y,
            road_type,
        ) = road_obs[i]

        if road_type == 0:  # road lane
            color = "lightgrey"
        elif road_type == 1:  # road line
            color = "grey"
        elif road_type == 2:  # road edge
            color = "black"
        ax.scatter(road_x, road_y, color=color, s=10)
        start = road_x + road_heading_x * road_length / 2
        end = road_x - road_heading_x * road_length / 2
        ax.plot(
            [start, end],
            [
                road_y + road_heading_y * road_length / 2,
                road_y - road_heading_y * road_length / 2,
            ],
            color=color,
            linewidth=1,
        )

    ax.axis((-1, 1, -1, 1))
    ax.set_aspect("equal", adjustable="box")

    return img_from_fig(fig)
