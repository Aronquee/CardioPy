"""FitzHugh-Nagumo cellular dynamics."""
from typing import Tuple, Union
import numpy as np

ArrayOrFloat = Union[float, np.ndarray]

# Default FHN model parameters (adjusted for an excitable medium)
DEFAULT_EPSILON = 0.01  # Recovery rate (the smaller, the slower)
DEFAULT_BETA = 0.7      # Polarization parameter
DEFAULT_GAMMA = 0.8     # Recovery parameter


def FHN_derivatives(
    V: ArrayOrFloat,
    w: ArrayOrFloat,
    I_stim: ArrayOrFloat,
    epsilon: float = DEFAULT_EPSILON,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> Tuple[ArrayOrFloat, ArrayOrFloat]:
    """
    Computes the derivatives of the FitzHugh-Nagumo system.

    Input:
        V (float or ndarray): Membrane potential
        w (float or ndarray): Recovery variable
        I_stim (float or ndarray): Applied stimulus current
        epsilon, beta, gamma (float): Model parameters -> Adjustable

    Return:
        dVdt, dwdt (tuple)
    """
    # Potential equation (fast) - Classic cubic term
    dVdt = V - (V ** 3) / 3 - w + I_stim

    # Recovery equation (slow)
    dwdt = epsilon * (V + beta - gamma * w)

    return dVdt, dwdt


def resting_state(
    V_resting: float = -1.0,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> Tuple[float, float]:
    """
    Resting values (I_stim = 0), solving V - V^3/3 - w = 0 and
    V + beta - gamma*w = 0 for the given V_resting.

    Return:
        (V_resting, w_resting)
    """
    w_resting = (V_resting + beta) / gamma
    return V_resting, w_resting