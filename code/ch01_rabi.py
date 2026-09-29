"""ch01_rabi.py -- Example A of Lecture 1: a resonantly driven qubit.

Simulates the Schroedinger equation  i d/dt |psi> = (w/2 sigma_z + u sigma_x) |psi>
for a constant drive u (Rabi oscillations) and writes two TikZ figures:
  figures/ch01_rabi_populations.tex   populations |<0|psi>|^2, |<1|psi>|^2 vs time
  figures/ch01_rabi_bloch.tex         the trajectory on the Bloch sphere
Run from the course root:  python3 code/ch01_rabi.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
import tikzexport as T

I, X, Y, Z = T.pauli()


def propagate(H, psi0, times):
    """Return the states psi(t) for a constant Hamiltonian H."""
    return [expm(-1j * H * t) @ psi0 for t in times]


omega, u = 1.0, 0.5                     # qubit frequency and drive amplitude
H_lab = omega / 2 * Z + u * X           # lab-frame Hamiltonian (off-resonant drive)
H_rot = u * X                           # rotating frame, on resonance
psi0 = np.array([1.0, 0.0], dtype=complex)   # start in |0>
times = np.linspace(0.0, 2 * np.pi / u, 400)

states_rot = propagate(H_rot, psi0, times)
states_lab = propagate(H_lab, psi0, times)
p1_rot = np.array([abs(s[1]) ** 2 for s in states_rot])
p1_lab = np.array([abs(s[1]) ** 2 for s in states_lab])

# --- populations vs time -------------------------------------------------
body = T.pgfplots_axis(
    [dict(x=times, y=p1_rot, label="resonant drive"),
     dict(x=times, y=p1_lab, label="detuned drive ($\\omega=2u$)", style="dashed")],
    xlabel="time $t$", ylabel="$|\\langle 1|\\psi(t)\\rangle|^2$",
    axis_options="ymin=0, ymax=1.05, legend pos=north east")
T.write_tikz("ch01_rabi_populations", body)

# --- Bloch-sphere trajectory ---------------------------------------------
def bloch(psi):
    rho = np.outer(psi, psi.conj())
    return T.bloch_vector(rho)

traj_rot = np.array([bloch(s) for s in states_rot[:200]])   # half a period: |0> -> |1>
traj_lab = np.array([bloch(s) for s in states_lab])
body = T.bloch_sphere([traj_rot, traj_lab],
                      points=[((0, 0, 1.0), ""), ((0, 0, -1.0), "")])
T.write_tikz("ch01_rabi_bloch", body, options="scale=0.9")
print("wrote ch01_rabi_populations.tex, ch01_rabi_bloch.tex")
