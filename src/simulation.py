"""Time-stepping drivers: single run, animation frames, and threshold search."""
import time
from typing import List, Tuple

import numpy as np

from .fhn_model import FHN_derivatives, resting_state
from .diffusion import apply_laplacian, implicit_diffusion_solver
from .stimulus import apply_stimulus


def simulate_tissue(
    method: str = 'explicit',
    I_amp: float = 10.0,
    t_dur: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    t_total: float = 50.0,
    dt: float = 0.01,
    D: float = 0.001,
    dx: float = 0.02,
    verbose: bool = True,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Main function that runs the 2D cardiac tissue simulation with FHN,
    using operator splitting (diffusion step + reaction step).

    Return:
        boundary_reached (bool): True if the wave reached the boundary.
        V (np.ndarray): Final potential state.
        w (np.ndarray): Final recovery state.
    """
    V_resting, w_resting = resting_state()
    V = np.full((Nx, Ny), V_resting)
    w = np.full((Nx, Ny), w_resting)

    center = (Nx // 2, Ny // 2)
    stim_radius = 3  # in number of nodes

    num_steps = int(t_total / dt)

    if verbose:
        print(f"Starting simulation: {method}, I_amp={I_amp}, t_dur={t_dur}ms")
    start_time = time.time()

    boundary_reached = False

    for step in range(num_steps):
        current_time = step * dt

        I_stim = apply_stimulus(V, center, stim_radius, I_amp, current_time, t_dur)

        if method == 'explicit':
            lap = apply_laplacian(V, dx)
            V = V + dt * D * lap
        elif method == 'implicit':
            V = implicit_diffusion_solver(V, dx, dt, D, theta=0.5)
        else:
            raise ValueError(f"Unknown method: {method!r} (use 'explicit' or 'implicit')")

        dV, dw = FHN_derivatives(V, w, I_stim)
        V = V + dt * dV
        w = w + dt * dw

        if step % 10 == 0:
            if (V[0, :] > 0.0).any() or (V[-1, :] > 0.0).any() \
               or (V[:, 0] > 0.0).any() or (V[:, -1] > 0.0).any():
                boundary_reached = True

    if verbose:
        print(f"Simulation time: {time.time() - start_time:.2f}s")
        print(f"Did the wave reach the boundary? {boundary_reached}")

    return boundary_reached, V, w


def simulate_and_animate(
    method: str = 'explicit',
    I_amp: float = 10.0,
    t_dur: float = 2.0,
    Nx: int = 60,
    Ny: int = 60,
    t_total: float = 50.0,
    dt: float = 0.02,
    D: float = 0.001,
    dx: float = 0.02,
    frame_interval: int = 20,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Runs the simulation and stores frames of potential V for animation.

    Return:
        frames (list): List of V matrices (frames).
        times (list): Times corresponding to each frame.
    """
    V_resting, w_resting = resting_state()
    V = np.full((Nx, Ny), V_resting)
    w = np.full((Nx, Ny), w_resting)
    center = (Nx // 2, Ny // 2)
    stim_radius = 3
    num_steps = int(t_total / dt)

    frames, times = [], []

    if verbose:
        print(f"Starting simulation with animation: {method}, I_amp={I_amp}, t_dur={t_dur}ms")
    start_time = time.time()

    for step in range(num_steps):
        current_time = step * dt

        I_stim = apply_stimulus(V, center, stim_radius, I_amp, current_time, t_dur)

        if method == 'explicit':
            lap = apply_laplacian(V, dx)
            V = V + dt * D * lap
        else:
            V = implicit_diffusion_solver(V, dx, dt, D, theta=0.5)

        dV, dw = FHN_derivatives(V, w, I_stim)
        V = V + dt * dV
        w = w + dt * dw

        if step % frame_interval == 0 or step == num_steps - 1:
            frames.append(V.copy())
            times.append(current_time)

    if verbose:
        print(f"Simulation completed in {time.time() - start_time:.2f}s")
        print(f"Frames saved: {len(frames)}")

    return frames, times


def find_threshold(
    t_dur: float,
    method: str = 'explicit',
    tol: float = 0.1,
    Nx: int = 60,
    Ny: int = 60,
    t_total: float = 80.0,
    dt: float = 0.02,
    max_iter: int = 20,
    I_low: float = 0.0,
    I_high: float = 20.0,
    verbose: bool = True,
) -> float:
    """
    Finds the threshold current for a given stimulus duration using
    binary search over I_amp.
    """
    if verbose:
        print(f"Searching for threshold for t_dur = {t_dur}ms...")

    for _ in range(max_iter):
        I_mid = (I_low + I_high) / 2
        success, _, _ = simulate_tissue(
            method=method, I_amp=I_mid, t_dur=t_dur,
            Nx=Nx, Ny=Ny, t_total=t_total, dt=dt, verbose=False,
        )
        if success:
            I_high = I_mid
        else:
            I_low = I_mid

        if (I_high - I_low) < tol:
            break

    I_threshold = (I_low + I_high) / 2
    if verbose:
        print(f"  -> Threshold found: {I_threshold:.3f} uA/cm²")
    return I_threshold