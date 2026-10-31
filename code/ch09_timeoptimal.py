"""ch09_timeoptimal.py -- time-optimal control of a qubit with drift.

    H(t) = (Delta/2) sigma_z + (u(t)/2) sigma_x,   |u(t)| <= u0,  |Delta| < u0.

The Pontryagin maximum principle gives a bang-bang control with a single
switching (Boscain et al.); the two bang durations are
    t1 = (pi - arccos(Delta^2/u0^2))/Omega,  t2 = (pi + arccos(Delta^2/u0^2))/Omega,
Omega = sqrt(u0^2 + Delta^2), total time 2*pi/Omega.  We integrate the
Schroedinger equation for the state and the adjoint state, evaluate the
switching function H_x(t) = Im <chi| sigma_x |psi> / 2, and export
  figures/ch09_timeoptimal_bloch.tex     Bloch-sphere trajectories
  figures/ch09_timeoptimal_controls.tex  control and switching function
"""
import sys, os
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

I2, SX, SY, SZ = T.pauli()
Delta, u0 = 0.6, 1.0
Omega = np.sqrt(u0**2 + Delta**2)


def propagate(u_of_t, times, psi0, chi0):
    """Integrate psi and the adjoint chi under H(t); return states and H_x(t)."""
    psi, chi = psi0.copy(), chi0.copy()
    psis, Hx = [psi.copy()], []
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        H = 0.5 * Delta * SZ + 0.5 * u_of_t(times[k]) * SX
        U = expm(-1j * H * dt)
        Hx.append(0.5 * np.imag(chi.conj() @ SX @ psi))   # switching function
        psi, chi = U @ psi, U @ chi                         # both obey Schroedinger
        psis.append(psi.copy())
    Hx.append(0.5 * np.imag(chi.conj() @ SX @ psi))
    return np.array(psis), np.array(Hx)


# --- bang-bang solution --------------------------------------------------
acs = np.arccos(Delta**2 / u0**2)
t1, t2 = (np.pi - acs) / Omega, (np.pi + acs) / Omega
tf = t1 + t2
bang = lambda t: u0 if t < t1 else -u0
times = np.linspace(0.0, tf, 801)
psi0 = np.array([1.0, 0.0], dtype=complex)
# adjoint initial condition: H_x(0)=0 is not required (we start mid-bang);
# choose chi(0) so that the switching happens at t1: chi(0) = -i sigma_x psi(0)
# rotated back; here we simply take chi(0) proportional to sigma_y psi(0)
# rotated so that H_x changes sign at t1 -- computed by a small search.
best = None
for phi in np.linspace(0, 2 * np.pi, 361):
    chi0 = np.cos(phi) * (SX @ psi0) + np.sin(phi) * (SY @ psi0)
    _, Hx = propagate(bang, times, psi0, chi0)
    k = np.argmin(np.abs(times - t1))
    score = abs(Hx[k]) - 0.2 * abs(Hx[0])
    if best is None or score < best[0]:
        best = (score, chi0)
chi0 = best[1]
psis, Hx = propagate(bang, times, psi0, chi0)
traj_bang = np.array([T.bloch_vector(np.outer(p, p.conj())) for p in psis])
fid_bang = abs(psis[-1][1])**2

# --- reference: drift-free geodesic (pi pulse), time pi/u0 -----------------
tg = np.linspace(0, np.pi / u0, 200)
traj_geo = np.array([[0.0, np.sin(u0 * s), np.cos(u0 * s)] for s in tg])

# --- reference: resonant pulse with the same drift but weaker amplitude ----
weak = lambda t: 0.5 * u0 * np.cos(Delta * t)   # slow Rabi drive on resonance
tw = np.linspace(0.0, 2.2 * tf, 1200)
psis_w, _ = propagate(weak, tw, psi0, chi0)
traj_weak = np.array([T.bloch_vector(np.outer(p, p.conj())) for p in psis_w])

T.write_tikz("ch09_timeoptimal_bloch",
             T.bloch_sphere([traj_bang, traj_geo, traj_weak],
                            points=[((0, 0, 1), "$\\ket{\\uparrow}$"),
                                    ((0, 0, -1), "$\\ket{\\downarrow}$")]))

uu = np.array([bang(t) for t in times])
T.write_tikz("ch09_timeoptimal_controls",
             T.pgfplots_axis([
                 dict(x=times, y=uu, label="$u(t)$ (bang-bang)", const=True),
                 dict(x=times, y=Hx / np.max(np.abs(Hx)), label="$H_x(t)$ (rescaled)", style="dashed"),
             ], xlabel="$t$", ylabel="", width="0.8\\linewidth", height="0.4\\linewidth",
                legend_pos="south west"))
print("t1=%.3f t2=%.3f tf=%.3f (2pi/Omega=%.3f) final population=%.4f"
      % (t1, t2, tf, 2 * np.pi / Omega, fid_bang))
