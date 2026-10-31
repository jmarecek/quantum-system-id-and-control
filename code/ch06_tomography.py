"""ch06_tomography.py -- the cost of tomography grows with the dimension.

The example state is the two-qubit system of Chapter 4 (ch04_coupling), both
qubits prepared in the tilted state x0 of Chapter 5 (ch05_spectrum), at the
time t = 0.5; it is shared by ch06_snapshots and ch06_shadows.  Pauli-by-Pauli
tomography measures each of the d^2 - 1 Pauli strings P on N/(d^2 - 1) copies,
estimates <P> by the mean of the outcomes +-1, and returns the linear inversion
    rho^ = (1 + sum_P <P>^ P) / d,    d = 2^q.
  (a) The 15 Pauli components of the pair, each a tile: the true value (tick)
      and the estimate from 100 shots (dot), with one standard deviation.
  (b) The trace distance (1/2)||rho^ - rho||_1, averaged over repetitions,
      against the total number of copies N, for q = 1 (the qubit of Chapter 5
      in x0), q = 2 (the pair), q = 3 (the pair and a qubit in x0), q = 4 (two
      pairs) and q = 5 (two pairs and a qubit): lines of slope -1/2, each
      further qubit moving the line to the right.
  (c) The copies needed for a trace distance of 0.05, against d (dots), and
      the law (4/(3 pi))^2 d^4/eps^2 (line).  The error matrix
      E = sum_P (<P>^ - <P>) P/d has squared Frobenius norm of order d^3/N;
      it is like a random Hermitian matrix, whose trace norm is about
      (8/(3 pi)) sqrt(d) ||E||_F, so that the trace distance is about
      (4/(3 pi)) d^2/sqrt(N) and the copies grow like d^4 (not d^5).  Lines
      of slope 3 (the best measurements of single copies) and 2 (the best
      joint measurements) through the first dot are drawn for comparison.
Writes figures/ch06_tomography.tex.
"""
import itertools

import numpy as np
from scipy.linalg import expm

import needham as N
import ch04_coupling as C4
from ch05_spectrum import X0
from ch06_shots import sample_mean

T_PAIR = 0.5                                     # time at which the pair is taken
PAULI = {"I": C4.I2, "X": C4.X, "Y": C4.Y, "Z": C4.Z}
QUBIT = (C4.I2 + X0[0] * C4.X + X0[1] * C4.Y + X0[2] * C4.Z) / 2      # the qubit of Chapter 5 in x0
EPS = 0.05
COLS = ["ndBlue", "ndGreen", "ndGold", "ndRed", "ndInk"]


def strings(q):
    """The 4^q - 1 Pauli strings on q qubits, without the identity, in the order IX.., ..., ZZ.."""
    return ["".join(s) for s in itertools.product("IXYZ", repeat=q)][1:]


def pauli(s):
    M = np.array([[1.0 + 0j]])
    for c in s:
        M = np.kron(M, PAULI[c])
    return M


def expectations(rho, names):
    return np.array([np.real(np.trace(pauli(s) @ rho)) for s in names])


def pair_state():
    """The two-qubit state of Chapter 4 at t = T_PAIR, both qubits prepared in x0."""
    rho0 = np.kron(QUBIT, QUBIT)
    x0 = np.append(expectations(rho0, strings(2)) / 2, 1.0)            # normalised basis P/2, then II
    x = expm(C4.model() * T_PAIR) @ x0
    rho = np.eye(4, dtype=complex) / 4
    for s, v in zip(strings(2), 2 * x[:15]):
        rho = rho + v * pauli(s) / 4
    return rho


def example(q):
    """The state on q qubits: pairs, and a qubit in x0 if q is odd."""
    rho = np.array([[1.0 + 0j]])
    for _ in range(q // 2):
        rho = np.kron(rho, pair_state())
    return np.kron(rho, QUBIT) if q % 2 else rho


def tomography_error(q, per_pauli, rng, reps):
    """Mean trace distance of Pauli-by-Pauli linear inversion with `per_pauli` shots per string."""
    names = strings(q)
    d = 2 ** q
    Ps = np.array([pauli(s) for s in names])
    c = expectations(example(q), names)
    out = []
    for _ in range(reps):
        dc = sample_mean(c, per_pauli, rng) - c
        E = np.tensordot(dc, Ps, axes=1) / d
        out.append(0.5 * np.abs(np.linalg.eigvalsh(E)).sum())
    return np.mean(out)


def tiles(b, rho, rng, shots=100, size=1.12, origin=(0.55, 0.0)):
    names = ["".join(s) for s in itertools.product("IXYZ", repeat=2)]
    ox, oy = origin
    for k, s in enumerate(names):
        i, j = divmod(k, 4)
        x0, y0 = ox + j * size, oy + (3 - i) * size
        if s == "II":
            b.append("\\draw[faint] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x0 + size, y0 + size))
            b.append("\\node[font=\\scriptsize, text=ndInk!60] at (%.3f,%.3f) {$II=1$};" % (x0 + size / 2, y0 + size / 2))
            continue
        v = np.real(np.trace(pauli(s) @ rho))
        est = sample_mean([v], shots, rng)[0]
        sd = np.sqrt(max(1 - est ** 2, 0.0) / shots)
        col = "ndSky" if v >= 0 else "ndRose"
        b.append("\\fill[%s, fill opacity=%.2f] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (col, 0.25 + 0.75 * min(abs(v) / 0.75, 1), x0, y0, x0 + size, y0 + size))
        b.append("\\draw[faint] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0, x0 + size, y0 + size))
        b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,%.3f) {$%s$};" % (x0 + size / 2, y0 + size - 0.03, s))
        # a little number line from -1 to 1
        lx = lambda u: x0 + 0.12 + (u + 1) / 2 * (size - 0.24)
        ly = y0 + 0.34
        b.append("\\draw[line width=0.3pt, ndInk!45] (%.3f,%.3f) -- (%.3f,%.3f);" % (lx(-1), ly, lx(1), ly))
        b.append("\\draw[line width=0.3pt, ndInk!45] (%.3f,%.3f) -- (%.3f,%.3f);" % (lx(0), ly - 0.05, lx(0), ly + 0.05))
        b.append("\\draw[line width=0.9pt, ndInk] (%.3f,%.3f) -- (%.3f,%.3f);" % (lx(v), ly - 0.13, lx(v), ly + 0.13))
        b.append("\\draw[line width=0.6pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (lx(est - sd), ly - 0.2, lx(est + sd), ly - 0.2))
        b.append("\\fill[ndRed] (%.3f,%.3f) circle (1.3pt);" % (lx(est), ly - 0.2))
        b.append("\\node[font=\\tiny, text=ndInk!80] at (%.3f,%.3f) {$%+.2f$};" % (x0 + size / 2, y0 + 0.6, v))
    for k, c in enumerate("IXYZ"):
        b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (ox - 0.22, oy + (3 - k) * size + size / 2, c))
        b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {$%s$};" % (ox + k * size + size / 2, oy + 4 * size + 0.2, c))
    b.append("\\node[font=\\scriptsize, rotate=90] at (%.3f,%.3f) {first qubit};" % (ox - 0.5, oy + 2 * size))
    b.append("\\node[font=\\scriptsize] at (%.3f,%.3f) {second qubit};" % (ox + 2 * size, oy + 4 * size + 0.5))


def main():
    rng = np.random.default_rng(3)
    rho = pair_state()
    assert abs(np.trace(rho) - 1) < 1e-12 and np.linalg.eigvalsh(rho).min() > -1e-12
    ev = dict(zip(strings(2), expectations(rho, strings(2))))
    assert abs(ev["ZI"] + 0.39) < 0.01 and abs(ev["XX"] - 0.45) < 0.01
    b = []
    # ---- (a) the 15 components of the pair
    tiles(b, rho, rng)
    b.append("\\node[note, anchor=north] at (2.79,-0.08) {tick: the true value; red: the\\\\ estimate from $100$ shots, $\\pm$ one s.d.};")
    b.append("\\node[font=\\small] at (2.79,-1.15) {(a) the pair: $15$ components};")
    # ---- (b) error against copies
    qs = [1, 2, 3, 4, 5]
    per = np.unique(np.logspace(1, 4, 7).astype(int))
    consts = []
    Pl = N.Plot((1.0, 7.3), (-3.3, 0.8), 4.6, 4.4, origin=(6.6, 0.0))
    b += Pl.axes("copies $N$", "trace distance", xticks=[(k, "$10^{%d}$" % k) for k in (2, 4, 6)],
                 yticks=[(k, "$10^{%d}$" % k) for k in (-3, -2, -1, 0)])
    for q, col in zip(qs, COLS):
        d = 2 ** q
        reps = 60 if q < 5 else 20
        Ns = per * (d * d - 1)
        err = np.array([tomography_error(q, n, rng, reps) for n in per])
        consts.append(np.median(err[2:] * np.sqrt(Ns[2:])))
        lx, ly = np.log10(Ns), np.log10(err)
        b.append(Pl.curve(lx, ly, "line width=0.9pt, %s" % col))
        b += ["\\fill[%s] %s circle (1.2pt);" % (col, Pl.pt(x, y)) for x, y in zip(lx, ly)]
        b.append("\\node[font=\\scriptsize, text=%s, anchor=south] at %s {$q=%d$};" % (col, Pl.pt(lx[0] - 0.1, ly[0] + 0.05), q))
        slope = np.polyfit(lx[2:], ly[2:], 1)[0]
        assert abs(slope + 0.5) < 0.06
        print("q = %d  d = %2d  trace distance * sqrt(N) = %.2f" % (q, d, consts[-1]))
    b.append("\\node[note, anchor=south west] at %s {each line falls like $N^{-1/2}$;\\\\ each qubit moves it\\\\ about $1.2$ decades on};" % Pl.pt(1.1, -3.25))
    b.append("\\node[font=\\small] at %s {(b) the error against copies};" % N.pt(Pl.P(4.15, -3.3) + np.array([0.0, -0.75])))
    # ---- (c) copies needed against d
    ds = 2.0 ** np.array(qs)
    need = (np.array(consts) / EPS) ** 2
    a = 2 * np.polyfit(np.log10(ds[2:]), np.log10(consts[2:]), 1)[0]          # q = 3, 4, 5
    assert 3.9 < a < 4.2
    law = (4 / (3 * np.pi)) ** 2 * ds ** 4 / EPS ** 2                          # trace distance (4/3pi) d^2/sqrt(N)
    assert np.all(np.abs(np.log(need[2:] / law[2:])) < 0.15)
    Q = N.Plot((0.15, 1.65), (2.5, 9.0), 4.2, 4.4, origin=(12.75, 0.0))
    b += Q.axes("$d=2^q$", "copies", xticks=[(np.log10(v), "$%d$" % v) for v in (2, 4, 8, 16, 32)],
                yticks=[(k, "$10^{%d}$" % k) for k in (3, 5, 7, 9)])
    lx = np.log10(ds)
    for p, col, lab in [(3, "ndGold", "$d^3$: best with\\\\ single copies"), (2, "ndGreen", "$d^2$: best with\\\\ joint measurements")]:
        b.append(Q.curve(lx, np.log10(need[0]) + p * (lx - lx[0]), "dashedink, %s" % col))
        b.append("\\node[note, text=%s, anchor=west] at %s {%s};" % (col, Q.pt(lx[-1] + 0.02, np.log10(need[0]) + p * (lx[-1] - lx[0]) - (0.25 if p == 3 else 0.0)), lab))
    b.append(Q.curve(lx, np.log10(law), "line width=0.9pt, ndRed"))
    b += ["\\node[dot, fill=ndRed] at %s {};" % Q.pt(x, y) for x, y in zip(lx, np.log10(need))]
    b.append("\\node[note, text=ndRed, anchor=south east] (f) at %s {Pauli by Pauli:\\\\ grows like $d^4$};" % Q.pt(1.15, 8.2))
    b.append("\\draw[pointer] (f.south) to[bend right=10] %s;" % Q.pt(1.28, np.log10((4 / (3 * np.pi)) ** 2 / EPS ** 2) + 4 * 1.28 + 0.1))
    b.append("\\node[font=\\small] at %s {(c) copies for an error of $%g$};" % (N.pt(Q.P(0.9, 2.5) + np.array([0.0, -0.75])), EPS))
    N.write("ch06_tomography", b)
    print("written figures/ch06_tomography.tex; exponent a = %.2f; copies for %.2f: %s" % (a, EPS, np.array2string(need, precision=2)))


if __name__ == "__main__":
    main()
