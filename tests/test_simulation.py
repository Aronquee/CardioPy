import numpy as np
import pytest

from src.simulation import simulate_tissue, simulate_and_animate, find_threshold


def test_simulate_tissue_returns_correct_shapes():
    boundary_reached, V, w = simulate_tissue(
        method='explicit', I_amp=5.0, t_dur=1.0,
        Nx=20, Ny=20, t_total=2.0, dt=0.02, verbose=False,
    )
    assert V.shape == (20, 20)
    assert w.shape == (20, 20)
    assert isinstance(boundary_reached, (bool, np.bool_))


def test_strong_stimulus_eventually_reaches_boundary():
    boundary_reached, V, _ = simulate_tissue(
        method='explicit', I_amp=15.0, t_dur=3.0,
        Nx=20, Ny=20, t_total=40.0, dt=0.02, verbose=False,
    )
    assert boundary_reached is True or boundary_reached == np.True_


def test_subthreshold_stimulus_does_not_propagate():
    boundary_reached, _, _ = simulate_tissue(
        method='explicit', I_amp=0.05, t_dur=0.5,
        Nx=20, Ny=20, t_total=10.0, dt=0.02, verbose=False,
    )
    assert not boundary_reached


def test_explicit_and_implicit_agree_qualitatively():
    # Same strong stimulus under both diffusion schemes should reach
    # the boundary within a similar timeframe (implicit allows larger dt).
    reached_exp, _, _ = simulate_tissue(
        method='explicit', I_amp=15.0, t_dur=3.0,
        Nx=20, Ny=20, t_total=40.0, dt=0.02, verbose=False,
    )
    reached_imp, _, _ = simulate_tissue(
        method='implicit', I_amp=15.0, t_dur=3.0,
        Nx=20, Ny=20, t_total=40.0, dt=0.05, verbose=False,
    )
    assert reached_exp == reached_imp


def test_simulate_and_animate_frame_count_and_times():
    frames, times = simulate_and_animate(
        method='explicit', I_amp=10.0, t_dur=1.0,
        Nx=15, Ny=15, t_total=1.0, dt=0.1, frame_interval=2, verbose=False,
    )
    assert len(frames) == len(times)
    assert all(f.shape == (15, 15) for f in frames)
    assert times == sorted(times)


def test_find_threshold_returns_positive_value_within_bounds():
    threshold = find_threshold(
        t_dur=2.0, method='explicit', tol=0.5,
        Nx=15, Ny=15, t_total=20.0, dt=0.05, max_iter=8, verbose=False,
    )
    assert 0.0 <= threshold <= 20.0


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        simulate_tissue(method='bogus', Nx=10, Ny=10, t_total=1.0, dt=0.1, verbose=False)