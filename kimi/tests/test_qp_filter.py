"""QP 安全过滤（加分项）：最小修改、精确验证、回退链路。"""
import numpy as np
import pytest

from safe_motion.config import RobotConfig
from safe_motion.geometry import Workspace
from safe_motion.qp_filter import QPFilter
from safe_motion.replay import run_scenario
from safe_motion.safety_filter import SafetyFilter

from conftest import SCENARIOS

CFG = RobotConfig.ur5()
WS = Workspace.from_dict({"x": [-0.7, 0.7], "y": [-0.7, 0.7], "z": [0.05, 0.9]})
DT = 0.05


def make_qp(**kw):
    return QPFilter(SafetyFilter(CFG, WS), **kw)


def test_qp_keeps_safe_action(real_01):
    """名义动作安全时 QP 不介入（原样执行）。"""
    f = make_qp()
    q = np.array(real_01["joint_state"])
    v_nom = f.nominal_velocity(q, np.array(real_01["action_chunk"])[0], DT)
    res = f.filter(q, v_nom, DT)
    assert res.status == "unmodified"
    np.testing.assert_allclose(res.q_dot, v_nom, atol=1e-12)


def test_qp_solution_safe_and_closer(real_01):
    """同一干预点（frame 25 → 49）：QP 解必须 (1) 精确验证安全，
    (2) 比速度缩放更接近名义速度（欧氏距离更小）。"""
    chunk = np.array(real_01["action_chunk"])
    scaling = SafetyFilter(CFG, WS)
    qp = make_qp()
    v_nom = scaling.nominal_velocity(chunk[25], chunk[49], DT)

    res_s = scaling.filter(chunk[25], v_nom, DT)
    res_q = qp.filter(chunk[25], v_nom, DT)

    ok, _, _ = qp.check_step(chunk[25], res_q.q_dot, DT)
    assert ok
    assert (np.linalg.norm(res_q.q_dot - v_nom)
            <= np.linalg.norm(res_s.q_dot - v_nom) + 1e-9)


def test_qp_replay_safe_and_less_deviation():
    """real_01 全程 QP：执行轨迹同样全程安全，且贴边滑行使
    TCP 偏离与停滞步数显著少于速度缩放（缩放：dev 0.186 m / 21 停）。"""
    r = run_scenario(SCENARIOS / "workspace_boundary.json",
                     filter_method="qp", make_plot=False)
    s = r["summary"]
    assert s["executed_steps"] == 50
    assert np.all(r["executed_margins"] >= -1e-6)
    assert s["max_tcp_deviation_m"] < 0.15
    assert s["stopped_steps"] < 21
    assert s["maximum_joint_velocity"] <= np.pi + 1e-9


def test_qp_falls_back_when_solver_fails(real_01, monkeypatch):
    """QP 求解器抛异常 → 回退速度缩放，结果仍然安全。"""
    import scipy.optimize as opt

    def boom(*a, **k):
        raise RuntimeError("solver down")

    monkeypatch.setattr(opt, "minimize", boom)
    chunk = np.array(real_01["action_chunk"])
    qp = make_qp()
    v_nom = qp.nominal_velocity(chunk[25], chunk[49], DT)
    res = qp.filter(chunk[25], v_nom, DT)
    assert "回退" in res.reason
    ok, _, _ = qp.check_step(chunk[25], res.q_dot, DT)
    assert ok


def test_qp_fail_closed_on_garbage(real_01, monkeypatch):
    monkeypatch.setattr(QPFilter, "filter",
                        lambda self, q, v, dt: (_ for _ in ()).throw(
                            RuntimeError("x")))
    qp = make_qp()
    res = qp.filter_fail_closed(np.array(real_01["joint_state"]),
                                np.ones(6), DT)
    assert res.status == "stopped"
    np.testing.assert_allclose(res.q_dot, np.zeros(6), atol=1e-15)
