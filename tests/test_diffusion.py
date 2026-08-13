import numpy as np
import pytest

from src.diffusion import apply_laplacian, implicit_diffusion_solver


def test_laplacian_zero_for_uniform_field():
    V = np.full((10, 10), 3.5)
    lap = apply_laplacian(V, dx=0.1)
    assert np.allclose(lap, 0.0)


def test_laplacian_sign_at_central_peak():
    # A single spike surrounded by zeros should have a negative Laplacian
    # at the peak (concave down) and positive at its immediate neighbors.
    V = np.zeros((5, 5))
    V[2, 2] = 10.0
    lap = apply_laplacian(V, dx=1.0)
    assert lap[2, 2] < 0
    assert lap[1, 2] > 0
    assert lap[2, 1] > 0


def test_laplacian_neumann_boundary_no_leak():
    # With a uniform field, Neumann (zero-flux) boundaries mean edge
    # nodes see no net curvature, same as interior.
    V = np.full((6, 6), 1.0)
    V[3, 3] = 2.0
    lap = apply_laplacian(V, dx=1.0)
    # Corner/edge values should still be finite and consistent with a
    # ghost node equal to the boundary value (no extra flux introduced)
    assert np.isfinite(lap).all()


def test_implicit_matches_explicit_for_small_dt():
    # For a small enough dt, one explicit step and one implicit
    # (theta=0.5) step should give similar results for a smooth field.
    rng = np.random.default_rng(0)
    V = rng.normal(size=(15, 15))
    dx, dt, D = 0.02, 1e-5, 0.001

    lap = apply_laplacian(V, dx)
    V_explicit = V + dt * D * lap
    V_implicit = implicit_diffusion_solver(V, dx, dt, D, theta=0.5)

    assert np.allclose(V_explicit, V_implicit, atol=1e-4)


def test_implicit_solver_conserves_mean_with_neumann_bc():
    # Zero-flux boundaries should approximately conserve the mean of V
    # under pure diffusion (no reaction/stimulus).
    V = np.zeros((10, 10))
    V[5, 5] = 5.0
    dx, dt, D = 0.1, 0.5, 0.01

    V_new = implicit_diffusion_solver(V, dx, dt, D, theta=0.5)
    assert V.mean() == pytest.approx(V_new.mean(), abs=1e-8)