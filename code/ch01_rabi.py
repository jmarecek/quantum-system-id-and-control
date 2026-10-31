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
import needham as N

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
p_max_lab = u ** 2 / (u ** 2 + omega ** 2 / 4)          # the largest population a detuned drive reaches
assert abs(p1_rot.max() - 1) < 1e-4 and abs(p1_lab.max() - p_max_lab) < 1e-3 and abs(p_max_lab - 0.5) < 1e-12

# --- populations vs time -------------------------------------------------
b = []
Pl = N.Plot((0.0, times[-1]), (0.0, 1.0), 9.0, 4.4)
b += Pl.axes("time $t$", "$\\abs{\\langle 1|\\psi(t)\\rangle}^2$",
             xticks=[(k * np.pi, "$%s$" % lab) for k, lab in [(1, "\\pi"), (2, "2\\pi"), (3, "3\\pi"), (4, "4\\pi")]],
             yticks=[(0.5, "$\\tfrac12$"), (1, "$1$")])
b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0, 1), Pl.pt(times[-1], 1)))
b.append("\\draw[faint, densely dotted] %s -- %s;" % (Pl.pt(0, p_max_lab), Pl.pt(times[-1], p_max_lab)))
b.append(Pl.curve(times, p1_rot, "line width=1.3pt, ndBlue"))
b.append(Pl.curve(times, p1_lab, "line width=1.1pt, ndGold, dash pattern=on 4pt off 2pt"))
t_pi = np.pi / (2 * u)                                    # the first maximum on resonance: a pi-pulse
b.append("\\node[dot, fill=ndBlue] at %s {};" % Pl.pt(t_pi, 1.0))
b.append("\\node[note, anchor=north, text=ndBlue] (r) at %s {resonant: all of\\\\ the population\\\\ reaches $\\ket1$ at\\\\ $t=\\pi/(2u)$};"
         % Pl.pt(2 * np.pi, 0.98))
b.append("\\draw[pointer] (r.north west) to[bend right=20] %s;" % Pl.pt(t_pi + 0.15, 1.0))
k = np.argmax(np.where(times > 3 * np.pi, p1_lab, 0))
b.append("\\node[note, anchor=south, text=ndGold!80!ndInk] (d) at %s {detuned, $\\omega=2u$: at most\\\\ $u^2/(u^2+\\omega^2/4)=\\tfrac12$};"
         % Pl.pt(times[k] - 0.6, 1.03))
b.append("\\draw[pointer] (d.south) to[bend left=10] %s;" % Pl.pt(times[k], p1_lab[k] + 0.03))
N.write("ch01_rabi_populations", b)

# --- Bloch-sphere trajectory ---------------------------------------------
def bloch(psi):
    rho = np.outer(psi, psi.conj())
    return T.bloch_vector(rho)

traj_rot = np.array([bloch(s) for s in propagate(H_rot, psi0, np.linspace(0, np.pi / (2 * u), 150))])   # |0> -> |1>
traj_lab = np.array([bloch(s) for s in states_lab])
assert np.allclose(traj_rot[-1], [0, 0, -1], atol=1e-3) and abs(traj_lab[:, 2].min()) < 5e-3

V = N.View(az=-50.0, el=16.0, R=2.0)
b = V.sphere()
b += V.great_circle([0, 0, 1])
for v, lab, pos in [((0, 0, 1.32), "z", "above"), ((1.45, 0, 0), "x", "below"), ((0, 1.2, 0), "y", "above left")]:
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
h_lab = np.array([2 * u, 0.0, omega]) / np.hypot(2 * u, omega)          # rotation axis of the detuned drive
b.append("\\draw[vec, ndBlue!70] %s -- %s;" % (V.pt((0, 0, 0)), V.pt((1, 0, 0))))
b.append("\\draw[vec, ndGold] %s -- %s;" % (V.pt((0, 0, 0)), V.pt(h_lab)))
b += V.curve(traj_rot, "line width=1.6pt, ndBlue", "line width=0.8pt, ndBlue, densely dotted")
b += V.curve(traj_lab, "line width=1.4pt, ndGold", "line width=0.8pt, ndGold, densely dotted")
k = len(traj_rot) // 3
b.append("\\draw[->, >=stealth, line width=1.6pt, ndBlue] %s -- %s;" % (V.pt(traj_rot[k]), V.pt(traj_rot[k + 3])))
b.append("\\node[dot, label={[font=\\footnotesize]above left:$\\ket0$}] at %s {};" % V.pt((0, 0, 1)))
b.append("\\node[dot, fill=ndBlue, label={[font=\\footnotesize]below:$\\ket1$}] at %s {};" % V.pt((0, 0, -1)))
low = traj_lab[np.argmin(traj_lab[:, 2])]
b.append("\\node[dot, fill=ndGold] at %s {};" % V.pt(low))
b.append("\\node[note, anchor=east, text=ndBlue] (a) at (%.2f,0.9) {resonant:\\\\ a great circle\\\\ about $x$, to $\\ket1$};" % (-V.R - 0.25))
b.append("\\draw[pointer] (a.east) to[bend left=15] %s;" % V.pt(traj_rot[int(0.62 * len(traj_rot))]))
b.append("\\node[note, anchor=west, text=ndGold!80!ndInk] (c) at (%.2f,-1.25) {detuned:\\\\ the axis tilts,\\\\ the circle stops\\\\ at the equator};" % (V.R + 0.3))
b.append("\\draw[pointer] (c.west) to[bend left=15] %s;" % V.pt(low))
N.write("ch01_rabi_bloch", b)
print("wrote ch01_rabi_populations.tex, ch01_rabi_bloch.tex")
