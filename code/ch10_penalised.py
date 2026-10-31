"""ch10_penalised.py -- penalised geodesics on SU(2) converging to a
sub-Riemannian geodesic as the penalty q -> infinity.

Model: right-invariant control system  dU/dt = -i (a_x sx + a_y sy + a_z sz) U
on SU(2).  The "cheap" (available) directions are D = span{sx, sz}; the
sy direction is penalised by q, i.e. the penalised energy is
    E_q(a) = 1/2 * int_0^1 ( a_x^2 + q a_y^2 + a_z^2 ) dt .
The target V = exp(-i theta sy/2) is a rotation about the penalised axis,
so the hard-constrained (sub-Riemannian) minimiser must generate sy through
the commutator [sx, sz].  We discretise a(t) as N piecewise-constant steps
and minimise E_q plus a stiff endpoint penalty with L-BFGS, warm-starting
each q from the previous solution.  Exports
  figures/ch10_penalised_bloch.tex     Bloch trajectories of U(t)|0> for several q
  figures/ch10_penalised_controls.tex  the penalised component a_y(t)
  figures/ch10_penalised_energy.tex    E_q and constraint violation versus q
"""
import sys, os
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

I2, SX, SY, SZ = T.pauli()
N = 24                      # time steps
dt = 1.0 / N
theta = np.pi / 2
V = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * SY   # target exp(-i theta sy/2)
psi0 = np.array([1.0, 0.0], dtype=complex)


def step(a):
    """exp(-i (a . sigma) dt) for a in R^3, closed form for 2x2."""
    r = np.linalg.norm(a)
    if r < 1e-14:
        return I2.copy()
    n = a / r
    return np.cos(r * dt) * I2 - 1j * np.sin(r * dt) * (n[0] * SX + n[1] * SY + n[2] * SZ)


def propagate(A):
    """A has shape (N,3); returns the list of propagators U_0=I, ..., U_N."""
    Us = [I2.copy()]
    for a in A:
        Us.append(step(a) @ Us[-1])
    return Us


def penalised_energy(A, q):
    return 0.5 * dt * np.sum(A[:, 0]**2 + q * A[:, 1]**2 + A[:, 2]**2)


def objective(x, q, mu):
    A = x.reshape(N, 3)
    UN = propagate(A)[-1]
    # endpoint mismatch up to a global phase is irrelevant in SU(2) here; use Frobenius
    viol = np.linalg.norm(UN - V) ** 2
    return penalised_energy(A, q) + mu * viol


def solve(q, x0, mu=2e3):
    res = minimize(objective, x0, args=(q, mu), method="L-BFGS-B",
                   options=dict(maxiter=4000, maxfun=400000))
    return res.x


rng = np.random.RandomState(0)
x = 0.3 * rng.randn(N * 3)
qs = [1, 4, 16, 64, 256]
sols, energies, viols, lengths = {}, [], [], []
for q in qs:
    # several starts: warm start from the previous q, plus random restarts;
    # the landscape has a spurious minimum that rotates directly about sy.
    starts = [x] + [0.6 * rng.randn(N * 3) for _ in range(3)]
    cands = [solve(q, s) for s in starts]
    x = min(cands, key=lambda c: objective(c, q, 2e3))
    A = x.reshape(N, 3)
    sols[q] = A
    Us = propagate(A)
    energies.append(penalised_energy(A, q))
    viols.append(dt * np.sum(A[:, 1] ** 2))                 # int g(P^perp a, P^perp a)
    lengths.append(dt * np.sum(np.sqrt(A[:, 0]**2 + A[:, 2]**2)))  # horizontal length
    print("q=%4d  E_q=%.4f  int a_y^2=%.2e  horizontal length=%.4f  |U_N-V|=%.1e"
          % (q, energies[-1], viols[-1], lengths[-1], np.linalg.norm(Us[-1] - V)))

# --- Bloch trajectories for three penalties ---------------------------------
trajs = []
for q in [1, 16, 256]:
    Us = propagate(sols[q])
    trajs.append(np.array([T.bloch_vector(np.outer(U @ psi0, (U @ psi0).conj())) for U in Us]))
T.write_tikz("ch10_penalised_bloch",
             T.bloch_sphere(trajs, points=[((0, 0, 1), "$U(0)\\ket0$"), ((1, 0, 0), "$V\\ket0$")]))

# --- penalised control component versus time -------------------------------
tt = np.arange(N) * dt
series = [dict(x=tt, y=sols[q][:, 1], label="$q=%d$" % q, const=True) for q in qs[:4]]
T.write_tikz("ch10_penalised_controls",
             T.pgfplots_axis(series, xlabel="$t$", ylabel="$a_y(t)$ (penalised direction)",
                             width="0.8\\linewidth", height="0.45\\linewidth", legend_pos="north east"))

# --- energies and violation versus q ---------------------------------------
T.write_tikz("ch10_penalised_energy",
             T.pgfplots_axis([dict(x=qs, y=energies, label="$\\min E_q$", style="mark=*"),
                              dict(x=qs, y=[q * v for q, v in zip(qs, viols)],
                                   label="$q\\int g(P^\\perp\\dot\\gamma_q,P^\\perp\\dot\\gamma_q)$", style="mark=square*, dashed")],
                             xlabel="penalty $q$", ylabel="", width="0.8\\linewidth",
                             height="0.45\\linewidth", legend_pos="north west",
                             axis_options="xmode=log, log basis x=2"))
