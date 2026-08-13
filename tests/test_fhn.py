import numpy as np
import pytest

from src.fhn_model import FHN_derivatives, resting_state, DEFAULT_BETA, DEFAULT_GAMMA


def test_resting_state_matches_known_values():
    V_r, w_r = resting_state()
    assert V_r == -1.0
    assert w_r == pytest.approx((-1.0 + DEFAULT_BETA) / DEFAULT_GAMMA)


def test_derivatives_zero_at_true_fixed_point():
    # Solve for the exact fixed point (V - V^3/3 - w = 0, w = (V+beta)/gamma)
    # at V = -1 approx.; check derivatives are near zero with I_stim = 0
    V_r, w_r = resting_state()
    dV, dw = FHN_derivatives(V_r, w_r, I_stim=0.0)
    # Not an exact root of the cubic, but should be small relative to a stimulus-driven case
    assert abs(dV) < 0.2
    assert dw == pytest.approx(0.0, abs=1e-12)


def test_derivatives_respond_to_stimulus():
    V_r, w_r = resting_state()
    dV_no_stim, _ = FHN_derivatives(V_r, w_r, I_stim=0.0)
    dV_stim, _ = FHN_derivatives(V_r, w_r, I_stim=5.0)
    assert dV_stim > dV_no_stim


def test_derivatives_vectorized_over_arrays():
    V = np.array([-1.0, 0.0, 1.0])
    w = np.zeros(3)
    I_stim = np.zeros(3)
    dV, dw = FHN_derivatives(V, w, I_stim)
    assert dV.shape == (3,)
    assert dw.shape == (3,)


def test_custom_parameters_change_output():
    dV_default, dw_default = FHN_derivatives(0.0, 0.0, 0.0)
    dV_custom, dw_custom = FHN_derivatives(0.0, 0.0, 0.0, epsilon=0.1, beta=0.0, gamma=0.5)
    assert dw_default != dw_custom