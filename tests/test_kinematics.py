import numpy as np

from safe_motion.kinematics import chain_points, forward_kinematics


def test_zero_configuration_matches_independent_known_translation():
    result = forward_kinematics(np.zeros(6))
    # Independently derived from the assignment DH table.
    np.testing.assert_allclose(result["tcp"], [-0.81725, -0.19145, -0.005491], atol=1e-6)


def test_chain_has_finite_base_six_joints_and_tcp():
    points = chain_points(np.array([0.2, -1.0, 1.2, -0.5, -1.2, 0.3]))
    assert points.shape == (8, 3)
    assert np.all(np.isfinite(points))
    np.testing.assert_allclose(points[-1], points[-2])
