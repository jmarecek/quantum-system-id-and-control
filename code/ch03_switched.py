"""ch03_switched.py -- control under noise as a switched system.

A qubit is driven by a pi pulse about the x-axis,
    H(t, xi) = u(t) sigma_x / 2 + b(t, xi) sigma_z / 2,
    u(t) = (2 pi / T) sin^2(pi t / T)          (area pi),
while the noise term switches at random times between the two modes
b = +b_0 and b = -b_0 (random telegraph noise from a two-level fluctuator,
switching rate kappa, stationary initial mode).  Every run is unitary: the
Bloch vector rotates, r' = h x r with h = (u, 0, b), and stays on the sphere.
The average over runs, the Bloch vector of E[U rho U^dagger], ends inside
the ball: the averaged map is a (mixed-unitary) quantum channel.
Writes figures/ch03_switched.tex.
"""
import numpy as np
import tikzexport as T
import needham as N

I, X, Y, Z = T.pauli()
TF = 2 * np.pi                   # duration of the pulse
BETA, KAPPA = 0.5, 0.4           # telegraph amplitude and switching rate
NSTEP = 800
TIMES = np.linspace(0, TF, NSTEP + 1)
DT = TF / NSTEP
N_RUNS = 4000
SEED = 7
COLS = ["ndBlue", "ndRed", "ndGold"]


def u(t):
    return (2 * np.pi / TF) * np.sin(np.pi * t / TF) ** 2


def telegraph(rng, m):
    """m runs of random telegraph noise on TIMES: values +-BETA, flips at rate KAPPA."""
    s = rng.choice([-1.0, 1.0], size=m)
    b = np.zeros((m, NSTEP + 1))
    flips = np.zeros(m, dtype=int)
    for k in range(NSTEP + 1):
        b[:, k] = BETA * s
        f = rng.random(m) < KAPPA * DT
        s = np.where(f, -s, s)
        flips += f & (k < NSTEP)
    return b, flips


def evolve(b):
    """Bloch vectors of the runs, from |0>, by exact rotations on each time step."""
    m = b.shape[0]
    r = np.zeros((m, NSTEP + 1, 3))
    r[:, 0] = [0.0, 0.0, 1.0]
    for k in range(NSTEP):
        h = np.stack([np.full(m, u(TIMES[k] + DT / 2)), np.zeros(m), b[:, k]], 1)
        nh = np.linalg.norm(h, axis=1)
        ax, ph = h / nh[:, None], nh * DT
        x = r[:, k]
        c, s = np.cos(ph)[:, None], np.sin(ph)[:, None]
        r[:, k + 1] = x * c + np.cross(ax, x) * s + ax * np.sum(ax * x, 1)[:, None] * (1 - c)
    return r


def check_unitary(bk, rk):
    """The same run as a product of 2x2 unitaries reproduces the Bloch path."""
    psi = np.array([1.0, 0.0], dtype=complex)
    for k in range(NSTEP):
        hv = np.array([u(TIMES[k] + DT / 2), 0.0, bk[k]])
        th = np.linalg.norm(hv) * DT
        n = hv / np.linalg.norm(hv)
        Uk = np.cos(th / 2) * I - 1j * np.sin(th / 2) * (n[0] * X + n[1] * Y + n[2] * Z)
        assert np.allclose(Uk.conj().T @ Uk, I)
        psi = Uk @ psi
    assert np.allclose(T.bloch_vector(np.outer(psi, psi.conj())), rk[-1], atol=1e-8)


def main():
    rng = np.random.default_rng(SEED)
    b, flips = telegraph(rng, N_RUNS)
    r = evolve(b)
    assert np.allclose(np.linalg.norm(r, axis=2), 1.0)          # every run stays on the sphere
    nominal = evolve(np.zeros((1, NSTEP + 1)))[0]
    assert np.allclose(nominal[-1], [0, 0, -1], atol=1e-6)       # the pi pulse alone: |0> -> |1>
    avg = r.mean(0)                                              # Bloch vector of E[U rho U^dagger]
    assert np.linalg.norm(avg[-1]) < 0.8                         # the average is mixed
    shown = [0, 1, 2]
    check_unitary(b[0], r[0])
    assert len(set(flips[shown])) == 3                           # three different switching patterns
    print("switches of the runs shown:", flips[shown], " |average| at T: %.3f" % np.linalg.norm(avg[-1]))

    out = []
    # ---- (a) timing diagram: the control, and the switching signal of three runs
    W, lane = 6.4, 1.15
    tops = [3.0 * lane, 2.0 * lane, 1.0 * lane, 0.0]
    Pu = N.Plot((0, TF), (0, 1.1), W, 0.85, origin=(0.0, tops[0]))
    uu = u(TIMES)
    out.append("\\fill[ndSky] %s -- %s -- cycle;" % (" -- ".join(Pu.pt(t, v) for t, v in zip(TIMES[::8], uu[::8])), Pu.pt(TF, 0)))
    out.append(Pu.curve(TIMES[::4], uu[::4], "ink"))
    out.append("\\draw[faint] %s -- %s;" % (Pu.pt(0, 0), Pu.pt(TF, 0)))
    out.append("\\node[anchor=east, font=\\small] at %s {$u(t)$};" % Pu.pt(0, 0.45))
    out.append("\\node[note, anchor=west] at %s {control: the same\\\\ $\\pi$ pulse in every run};" % Pu.pt(TF * 1.03, 0.45))
    for j, k in enumerate(shown):
        Pb = N.Plot((0, TF), (-BETA, BETA), W, 0.62, origin=(0.0, tops[j + 1] + 0.1))
        out.append("\\draw[faint] %s -- %s;" % (Pb.pt(0, 0), Pb.pt(TF, 0)))
        out.append(Pb.curve(TIMES, b[k], "line width=1.1pt, %s" % COLS[j]))
        out.append("\\node[anchor=east, font=\\small, text=%s] at %s {run %d};" % (COLS[j], Pb.pt(0, 0), j + 1))
    Pb = N.Plot((0, TF), (-BETA, BETA), W, 0.62, origin=(0.0, tops[1] + 0.1))
    out.append("\\node[font=\\scriptsize, anchor=west] at %s {$+b_0$};" % Pb.pt(TF * 1.01, BETA))
    out.append("\\node[font=\\scriptsize, anchor=west] at %s {$-b_0$};" % Pb.pt(TF * 1.01, -BETA))
    Pb3 = N.Plot((0, TF), (-BETA, BETA), W, 0.62, origin=(0.0, tops[2] + 0.1))
    out.append("\\node[note, anchor=west] at %s {noise: $\\pm\\tfrac{b_0}{2}\\sigz$,\\\\ switching at random\\\\ times, different\\\\ in every run};" % Pb3.pt(TF * 1.1, 0))
    out.append("\\draw[faint, ->, >=stealth] (0,-0.25) -- (%.2f,-0.25) node[right, text=ndInk] {$t$};" % (W + 0.25))
    out.append("\\draw[faint] (%.2f,-0.19) -- (%.2f,-0.31) node[below, font=\\scriptsize, text=ndInk] {$T$};" % (W, W))
    out.append("\\draw[faint] (0,-0.19) -- (0,-0.31) node[below, font=\\scriptsize, text=ndInk] {$0$};")
    out.append("\\node[font=\\small] at (%.2f,-1.4) {(a) control and switching signals};" % (W / 2))

    # ---- (b) the runs on the Bloch sphere
    V = N.View(az=-60.0, el=15.0, R=2.3, shift=(13.6, 1.7))
    out += V.sphere()
    out += V.great_circle([0, 0, 1])
    for v, lab, pos in [((0, 0, 1.22), "\\ket0", "above"), ((1.5, 0, 0), "x", "below")]:
        out.append("\\draw[faint, ->, >=stealth] %s -- %s node[%s, text=ndInk] {$%s$};" % (V.pt((0, 0, 0)), V.pt(v), pos, lab))
    out += V.curve(nominal, "dashedink, line width=0.9pt", "dashedink, densely dotted")
    for j, k in enumerate(shown):
        out += V.curve(r[k], "line width=1.3pt, %s" % COLS[j], "line width=0.6pt, %s, densely dotted" % COLS[j])
        out.append("\\node[dot, fill=%s] at %s {};" % (COLS[j], V.pt(r[k][-1])))
    out.append(N.polyline([V.P(p) for p in avg[::4]], "line width=1.6pt, ndGreen"))
    out.append("\\node[dot, fill=ndGreen, minimum size=4.6pt] at %s {};" % V.pt(avg[-1]))
    out.append("\\node[dot, fill=ndInk] at %s {};" % V.pt((0, 0, 1)))
    out.append("\\node[font=\\small, below=3pt] at %s {$\\ket1$};" % V.pt((0, 0, -1)))
    x0 = V.shift[0] + V.R + 0.5
    out.append("\\node[note, anchor=west] (n) at (%.2f,%.2f) {without noise (dashed):\\\\ $\\ket0\\to\\ket1$};" % (x0, V.shift[1] + 1.9))
    out.append("\\draw[pointer] (n.west) to[bend right=12] %s;" % V.pt(nominal[NSTEP // 3]))
    out.append("\\node[note, anchor=west] (s) at (%.2f,%.2f) {each run is unitary:\\\\ it stays on the sphere,\\\\ but on its own path};" % (x0, V.shift[1] + 0.2))
    out.append("\\draw[pointer] (s.west) to[bend left=10] %s;" % V.pt(r[shown[0]][int(0.7 * NSTEP)]))
    out.append("\\node[note, anchor=west, text=ndGreen!60!black] (a) at (%.2f,%.2f) {average over %d runs:\\\\ leaves the sphere,\\\\ the state is mixed};" % (x0, V.shift[1] - 1.7, N_RUNS))
    out.append("\\draw[pointer] (a.west) to[bend left=12] %s;" % V.pt(avg[-1]))
    out.append("\\node[font=\\small] at (%.2f,-1.4) {(b) the runs on the Bloch sphere};" % V.shift[0])
    N.write("ch03_switched", out)
    print("written figures/ch03_switched.tex")


if __name__ == "__main__":
    main()
