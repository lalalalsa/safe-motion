"""3D 可视化：矩形工作空间 + 机械臂骨架 + VLA 原始轨迹 vs 实际轨迹。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .kinematics import forward_kinematics


def _tcp_of(qs):
    """把关节轨迹转成 TCP 笛卡尔轨迹 (N, 3)。"""
    return np.array([forward_kinematics(q)[-1] for q in np.asarray(qs)])


def _draw_workspace(ax, ws, alpha=0.08, color="gray"):
    x0, x1 = ws["x"]
    y0, y1 = ws["y"]
    z0, z1 = ws["z"]
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # 底面
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # 顶面
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # 前
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # 右
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],  # 后
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],  # 左
    ]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor="k", linewidths=0.5)
    ax.add_collection3d(poly)


def _draw_arm(ax, q, color, alpha=0.85, lw=2.5, label=None):
    pts = forward_kinematics(q)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=color, lw=lw,
            markersize=3, alpha=alpha, label=label)


def plot_replay(scenario, report, out_path):
    ws = scenario["workspace"]
    nominal_tcp = _tcp_of(report["nominal_trajectory"])
    executed_tcp = _tcp_of(report["executed_trajectory"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    _draw_workspace(ax, ws)

    _draw_arm(ax, report["executed_trajectory"][0], "royalblue", label="arm @ start")
    _draw_arm(ax, report["executed_trajectory"][-1], "purple", label="arm @ end")

    ax.plot(nominal_tcp[:, 0], nominal_tcp[:, 1], nominal_tcp[:, 2],
            "-", color="red", lw=1.5, label="VLA nominal TCP")
    ax.plot(executed_tcp[:, 0], executed_tcp[:, 1], executed_tcp[:, 2],
            "-", color="green", lw=2.5, label="executed TCP")

    # 坐标轴略大于 workspace，使越界的原始轨迹可见
    pad = 0.15
    ax.set_xlim(ws["x"][0] - pad, ws["x"][1] + pad)
    ax.set_ylim(ws["y"][0] - pad, ws["y"][1] + pad)
    ax.set_zlim(ws["z"][0] - pad, ws["z"][1] + pad)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{report['name']}\nmodified={report['modified_steps']} "
                 f"stopped={report['stopped_steps']} "
                 f"min_margin={report['minimum_workspace_margin']:.3f}m")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
