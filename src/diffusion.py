"""2D spatial diffusion: explicit Laplacian and implicit (Crank-Nicolson) solver."""
import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import splu


def apply_laplacian(V: np.ndarray, dx: float) -> np.ndarray:
    """
    Computes the 2D Laplacian with Neumann boundary conditions
    (zero flux) at the boundaries using a reflected ghost node: V_ghost = V_boundary.
    If the values are equal, there is no current.

    The boundary node continues receiving current from the internal
    neighbor (it remains coupled to the rest of the tissue), but it does
    not allow current to "leak" outside the domain.

    Input:
        V (np.ndarray 2D): Potential matrix.
        dx (float): Grid spacing (cm).

    Return:
        lap (np.ndarray 2D): Laplacian matrix (dV/dx² + dV/dy²).
    """
    V_pad = np.pad(V, pad_width=1, mode='edge')
    V_up = V_pad[:-2, 1:-1]
    V_down = V_pad[2:, 1:-1]
    V_left = V_pad[1:-1, :-2]
    V_right = V_pad[1:-1, 2:]

    lap = (V_up + V_down + V_left + V_right - 4 * V) / (dx ** 2)
    return lap


# Cache for the matrices/factorization of the implicit method.
# Building and factorizing the sparse matrix is expensive; since
# (Nx, Ny, dx, dt, D, theta) do not change during a simulation, we do
# this once per parameter combination and reuse it at every time step.
_implicit_cache = {}


def _laplacian_1d_neumann(n: int, d: float):
    """
    1D second-derivative operator (n x n) with Neumann boundary conditions
    (zero flux) at both ends, using a reflected ghost node (same
    convention used in `apply_laplacian`).
    """
    principal = -2.0 * np.ones(n)
    neighbor = np.ones(n - 1)
    L = diags([neighbor, principal, neighbor], offsets=[-1, 0, 1], format='lil') / d ** 2
    # At the ends, the "ghost neighbor" has the same value as the boundary
    # node, so its contribution cancels part of -2 -> becomes -1
    L[0, 0] = -1.0 / d ** 2
    L[-1, -1] = -1.0 / d ** 2
    return L.tocsr()


def _build_2d_laplacian_operator(Nx: int, Ny: int, dx: float):
    """
    Builds the 2D Laplacian operator (Nx*Ny x Nx*Ny) in sparse format via
    the Kronecker sum of the 1D operators (x and y), consistent with
    `apply_laplacian` (Neumann/zero flux at the boundaries).
    Flatten in C-order: index k = i*Ny + j.
    """
    Lx = _laplacian_1d_neumann(Nx, dx)
    Ly = _laplacian_1d_neumann(Ny, dx)
    Ix = identity(Nx, format='csr')
    Iy = identity(Ny, format='csr')

    return kron(Lx, Iy, format='csc') + kron(Ix, Ly, format='csc')


def implicit_diffusion_solver(
    V: np.ndarray, dx: float, dt: float, D: float, theta: float = 0.5
) -> np.ndarray:
    """
    Solves 2D diffusion using Implicit FDM (Crank-Nicolson).

    Equation: (I - theta*dt*D*L) V_new = (I + (1-theta)*dt*D*L) V_old

    Input:
        V (np.ndarray 2D): Potential matrix at the current time.
        dx (float): Grid spacing.
        dt (float): Time step.
        D (float): Diffusion coefficient.
        theta (float): Scheme parameter (0.5 = Crank-Nicolson).

    Return:
        V_new (np.ndarray 2D): Matrix after implicit diffusion.
    """
    Nx, Ny = V.shape
    N = Nx * Ny

    key = (Nx, Ny, dx, dt, D, theta)
    if key not in _implicit_cache:
        L = _build_2d_laplacian_operator(Nx, Ny, dx)
        I = identity(N, format='csc')
        A = (I - theta * dt * D * L).tocsc()
        M = (I + (1 - theta) * dt * D * L).tocsc()
        lu = splu(A)
        _implicit_cache[key] = (M, lu)

    M, lu = _implicit_cache[key]

    b = M @ V.flatten()
    V_new = lu.solve(b).reshape(Nx, Ny)
    return V_new