"""可视化：3D 工作空间图 + margin 曲线图（面试演示素材）。"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 默认无头后端；show=True 时由 pyplot 弹窗
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .kinematics import forward_kinematics


def _tcp_path(qs):
    return np.array([forward_kinematics(q)["tcp"] for q in np.asarray(qs)])


def _draw_workspace(ax, ws, alpha=0.07):
    x0, x1, y0, y1, z0, z1 = ws.xmin, ws.xmax, ws.ymin, ws.ymax, ws.zmin, ws.zmax
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    ]
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor="gray", edgecolor="k", linewidths=0.6))


def _draw_arm(ax, q, color, alpha, lw, label=None):
    pts = forward_kinematics(q)["points"]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=color, alpha=alpha,
            lw=lw, markersize=3, label=label)


def _node_traces(qs):
    """(N, 6) 关节轨迹 → (N, 7, 3) 全部节点坐标。"""
    return np.array([forward_kinematics(q)["points"] for q in np.asarray(qs)])


def _draw_boundary_plane(ax, ws, boundary, lo, hi):
    """把最关键的那条边界画成红色半透明平面，横穿当前视野。"""
    if boundary in ("z_min", "z_max"):
        z = ws.zmin if boundary == "z_min" else ws.zmax
        quad = [(lo[0], lo[1], z), (hi[0], lo[1], z),
                (hi[0], hi[1], z), (lo[0], hi[1], z)]
    elif boundary in ("x_min", "x_max"):
        x = ws.xmin if boundary == "x_min" else ws.xmax
        quad = [(x, lo[1], lo[2]), (x, hi[1], lo[2]),
                (x, hi[1], hi[2]), (x, lo[1], hi[2])]
    else:
        y = ws.ymin if boundary == "y_min" else ws.ymax
        quad = [(lo[0], y, lo[2]), (hi[0], y, lo[2]),
                (hi[0], y, hi[2]), (lo[0], y, hi[2])]
    ax.add_collection3d(Poly3DCollection(
        [quad], alpha=0.13, facecolor="red", edgecolor="red",
        linewidths=1.4, linestyles="--"))


def plot_3d(out_path, name, ws, action_chunk, executed, records, show=False):
    """轨迹活动区视野 + 机械臂 + 名义（红）vs 实际（绿）TCP +
    最差边界平面（红色半透明）+ 最差非 TCP 节点的名义/实际轨迹。"""
    nom_pts = _node_traces(action_chunk)      # (50, 7, 3)
    exe_pts = _node_traces(executed)          # (51, 7, 3)
    nom_tcp, exe_tcp = nom_pts[:, 6], exe_pts[:, 6]

    # 找全局最差点：名义轨迹上哪个节点、越哪条边界
    node_m = np.stack([ws.margins(nom_pts[:, i, :]) for i in range(1, 7)],
                      axis=1)                    # (50, 6)
    f_worst, n_worst = np.unravel_index(int(np.argmin(node_m)), node_m.shape)
    worst_node = n_worst + 1
    worst_boundary = ws.boundary_name(nom_pts[f_worst, worst_node])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_workspace(ax, ws)

    _draw_arm(ax, executed[0], "royalblue", 0.9, 2.0, "arm @ start")
    _draw_arm(ax, executed[-1], "purple", 0.9, 2.0, "arm @ end")

    ax.plot(nom_tcp[:, 0], nom_tcp[:, 1], nom_tcp[:, 2], "-", color="red",
            lw=1.5, label="VLA nominal TCP")
    ax.plot(exe_tcp[:, 0], exe_tcp[:, 1], exe_tcp[:, 2], "-", color="green",
            lw=2.5, label="executed TCP")

    # 视野聚焦到轨迹活动区（含最差节点轨迹与首尾位形）；被越的边界平面
    # 天然落在名义（越界侧）与实际（安全侧）极值之间，必定横穿视野。
    pad = 0.15
    focus = np.vstack([nom_tcp, exe_tcp,
                       nom_pts[:, worst_node], exe_pts[:, worst_node],
                       forward_kinematics(executed[0])["points"],
                       forward_kinematics(executed[-1])["points"]])
    lo, hi = focus.min(axis=0) - pad, focus.max(axis=0) + pad

    _draw_boundary_plane(ax, ws, worst_boundary, lo, hi)

    # 最差点不是 TCP 时（如 real_03 的 joint_3），额外画该节点的两条轨迹
    if worst_node != 6:
        ax.plot(nom_pts[:, worst_node, 0], nom_pts[:, worst_node, 1],
                nom_pts[:, worst_node, 2], "--", color="darkorange", lw=1.5,
                label=f"joint_{worst_node} nominal")
        ax.plot(exe_pts[:, worst_node, 0], exe_pts[:, worst_node, 1],
                exe_pts[:, worst_node, 2], "-.", color="teal", lw=2.0,
                label=f"joint_{worst_node} executed")

    # 干预点标记：modified=橙，stopped=黑
    for status, color, marker, label in (
            ("modified", "orange", "D", "modified step"),
            ("stopped", "black", "x", "stopped step")):
        idx = [r["index"] for r in records if r["status"] == status]
        if idx:
            pts = exe_tcp[np.array(idx) + 1]  # 干预后到达的 TCP
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker=marker,
                       s=45, label=label, zorder=5)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(max(0.0, lo[2] - 0.05), hi[2])
    ax.view_init(elev=18, azim=-128)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    n_mod = sum(1 for r in records if r["status"] == "modified")
    n_stop = sum(1 for r in records if r["status"] == "stopped")
    ax.set_title(f"{name}  (modified={n_mod}, stopped={n_stop})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_margins(out_path, name, nominal_margins, executed_margins):
    """margin 曲线：未过滤 VLA（红）vs 实际执行（绿），零线为安全边界。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0.0, color="k", lw=1.0, ls="--", label="workspace boundary")
    ax.plot(range(len(nominal_margins)), nominal_margins, "r-o", ms=3, lw=1.2,
            label="VLA nominal (unfiltered)")
    ax.plot(range(len(executed_margins)), executed_margins, "g-o", ms=3, lw=1.6,
            label="executed (SafeMotion)")
    ax.fill_between(range(len(nominal_margins)), nominal_margins, 0.0,
                    where=(nominal_margins < 0), color="red", alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("min full-body margin (m)")
    ax.set_title(f"{name} — full-body workspace margin per step")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
