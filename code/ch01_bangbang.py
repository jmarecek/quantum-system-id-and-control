"""ch01_bangbang.py -- Question A: a resonant pi-pulse versus the bang-bang
(time-optimal) solution for a qubit with drift and a bounded control.
H = (omega/2) sigma_z + u(t) sigma_x, |u| <= u0.  Writes figures/ch01_bangbang.tex."""
import os, sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

I, X, Y, Z = T.pauli()
omega, u0 = 1.0, 0.4
def evolve(us, dt, psi0):
    psi = psi0.copy(); out = [psi0]
    for u in us:
        H = 0.5 * omega * Z + u * X
        psi = expm(-1j * H * dt) @ psi; out.append(psi)
    return np.array(out)
psi0 = np.array([1, 0], dtype=complex)
# (a) resonant drive in the lab frame, u(t) = 2 u0 cos(omega t): rotating-frame amplitude u0, flip after T = pi/(2 u0)
Ta = np.pi / (2 * u0); Na = 600; ta = np.linspace(0, Ta, Na + 1); ua = u0 * np.cos(omega * ta[:-1]) * 2  # 2u0 cos = rotating-frame amplitude u0
pa = evolve(ua, Ta / Na, psi0); popa = np.abs(pa[:, 1]) ** 2
# (b) bang-bang: constant u = +u0 for the whole time; with drift this only reaches sin^2 of the tilted axis,
# so the time-optimal solution switches sign a few times; we search over 2 switching times.
best = None
Tb = 1.4 * np.pi / (2 * u0)
for n1 in range(5, 60, 2):
    for n2 in range(5, 60, 2):
        seq = [u0] * n1 + [-u0] * n2 + [u0] * n1
        dt = 0.05
        pb = evolve(seq, dt, psi0); f = np.abs(pb[-1, 1]) ** 2
        if best is None or f > best[0]:
            best = (f, seq, dt)
f, seq, dt = best
tb = np.arange(len(seq) + 1) * dt
pb = evolve(seq, dt, psi0); popb = np.abs(pb[:, 1]) ** 2
ctrl = T.pgfplots_axis([dict(x=ta[:-1], y=ua, label=r"resonant $u(t)=2u_0\cos\omega t$"),
                        dict(x=tb[:-1], y=seq, label=r"bang-bang, $|u|=u_0$", const=True, style="dashed")],
                       xlabel="$t$", ylabel="$u(t)$", width="0.48\\linewidth", height="0.32\\linewidth", legend_pos="south east")
pops = T.pgfplots_axis([dict(x=ta, y=popa, label="resonant"),
                        dict(x=tb, y=popb, label="bang-bang", style="dashed")],
                       xlabel="$t$", ylabel=r"$|\langle 1|\psi(t)\rangle|^2$", width="0.48\\linewidth", height="0.32\\linewidth", legend_pos="south east")
T.write_tikz("ch01_bangbang_controls", ctrl)
T.write_tikz("ch01_bangbang_populations", pops)
print("written figures/ch01_bangbang_*.tex  resonant T=%.2f final=%.3f  bang-bang T=%.2f final=%.3f" % (Ta, popa[-1], tb[-1], f))
