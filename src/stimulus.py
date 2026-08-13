"""Electrode stimulus application."""
from typing import Tuple
import numpy as np


def apply_stimulus(
    V: np.ndarray,
    center: Tuple[int, int],
    radius_nodes: int,
    I_amp: float,
    t: float,
    t_dur: float,
) -> np.ndarray:
    """
    Adds the stimulus current to the central circular region.

    Input:
        V (np.ndarray): Current potential matrix (used only for shape).
        center (tuple): (i_center, j_center).
        radius_nodes (int): Electrode radius in number of nodes.
        I_amp (float): Current amplitude (uA/cm²).
        t (float): Current time (ms).
        t_dur (float): Pulse duration (ms).

    Return:
        I_stim_matrix (np.ndarray): Matrix with the current injected at each node.
    """
    I_stim = np.zeros_like(V)

    x = np.arange(V.shape[0])
    y = np.arange(V.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    if t <= t_dur:
        I_stim[dist <= radius_nodes] = I_amp

    return I_stim