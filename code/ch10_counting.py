"""ch10_counting.py -- how many gates does a unitary need?

(a), (b) The gate set {H, T} on one qubit, up to a global phase, that is as
rotations of the Bloch sphere: H is the half-turn about (x + z)/sqrt(2) and T
the eighth of a turn about z.  We enumerate all products of at most L of them,
removing repetitions as we go, and record
  (a) the number N_L of distinct rotations, against the 2^{L+1} - 1 words; it
      grows by a factor of about 1.30 for each further gate (N_36 = 290 787);
  (b) how well they cover the rotation group: for 20 000 rotations V drawn from
      the Haar measure, the angle of the rotation W^{-1} V for the nearest W
      (a phase-insensitive distance), its largest and its mean value, against
      the counting bound eps >= (6 pi / N_L)^{1/3}: N_L balls of angular radius
      eps, each of volume about (4 pi/3)(eps/2)^3 in the unit 3-sphere, must cover
      the rotation group, of volume pi^2.  The distance falls exponentially with
      L, so that log(1/eps) grows in proportion to the length of the word.  The
      largest distance is between two and three times the bound and the mean
      about equal to it (0.108 and 0.040 against 0.040 at L = 36).
  (c) The counting argument for n qubits.  A circuit of L gates drawn from g
      gates on one or two qubits is one of at most (g n^2)^L, while
      covering SU(2^n), of dimension 4^n - 1, to the precision eps takes at least
      about (1/eps)^{4^n - 1} balls.  Hence most unitaries need at least
          L >= (4^n - 1) log(1/eps) / log(g n^2)
      gates, plotted with g = 3 (H, T and CNOT) for eps = 10^-2 and 10^-6.  It
      grows like 4^n: for 20 qubits and eps = 10^-2 it is 7.1 * 10^11.
Writes figures/ch10_counting.tex.
"""
import numpy as np
from scipy.spatial import cKDTree
import needham as N

LMAX = 36
HQ = np.array([0.0, 1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)])                 # quaternion of H, up to phase
TQ = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])            # quaternion of T, up to phase
G_GATES = 3


def qmul(a, b):
    """Quaternion products a_i * b for the rows a_i of a."""
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def keys(q):
    """A key for q up to sign: make the first component of size > 1e-9 positive, then round."""
    lead = np.argmax(np.abs(q) > 1e-9, axis=1)
    s = np.sign(q[np.arange(len(q)), lead])
    return [tuple(k) for k in np.round(q * s[:, None] * 1e8).astype(np.int64)]


def enumerate_words(lmax=LMAX):
    """Distinct rotations reached by words of length <= L, for L = 0..lmax (breadth first)."""
    level = np.array([[1.0, 0, 0, 0]])
    seen = set(keys(level))
    layers, counts = [level], [1]
    for _ in range(lmax):
        new = np.vstack([qmul(level, HQ), qmul(level, TQ)])
        fresh = {}
        for i, k in enumerate(keys(new)):
            if k not in seen and k not in fresh:
                fresh[k] = i
        level = new[list(fresh.values())]
        seen.update(fresh)
        layers.append(level)
        counts.append(len(seen))
    return layers, np.array(counts)


def covering(layers, ls, rng=np.random.default_rng(0), m=20000):
    targets = rng.normal(size=(m, 4))
    targets /= np.linalg.norm(targets, axis=1)[:, None]
    out = []
    for L in ls:
        Q = np.vstack(layers[:L + 1])
        d, _ = cKDTree(np.vstack([Q, -Q])).query(targets)
        ang = 4 * np.arcsin(np.clip(d / 2, 0, 1))                       # rotation angle of W^{-1} V
        out.append((ang.max(), ang.mean()))
    return np.array(out)


def bound_eps(n_rot):
    return (6 * np.pi / n_rot) ** (1 / 3)


def lower_bound_gates(n, eps, g=G_GATES):
    return (4.0 ** n - 1) * np.log(1 / eps) / np.log(g * n ** 2 if n > 1 else g)


def checks(layers, counts, cov, ls):
    # the words are rotations: unit quaternions, and H, T have the expected orders (H^2 = T^8 = 1 up to phase)
    Q = np.vstack(layers)
    assert np.allclose(np.linalg.norm(Q, axis=1), 1)
    h2 = qmul(HQ[None], HQ)[0]
    t8 = TQ[None]
    for _ in range(7):
        t8 = qmul(t8, TQ)
    assert np.allclose(np.abs(h2[0]), 1) and np.allclose(np.abs(t8[0, 0]), 1)
    # the counts: far fewer than the number of words, growing by about 1.3 per gate
    assert counts[36] == 290787
    ratio = counts[LMAX] / counts[LMAX - 1]
    assert 1.25 < ratio < 1.35
    # the distances fall with L and stay above (largest) and near (mean) the counting bound
    assert np.all(cov[:, 0] >= bound_eps(counts[ls]) * 0.999)
    assert np.all(np.diff(cov[:, 1]) < 0)
    assert 0.5 < cov[-1, 1] / bound_eps(counts[ls[-1]]) < 1.5
    # the counting bound for n qubits grows like 4^n
    assert 7e11 < lower_bound_gates(20, 1e-2) < 7.2e11


def main():
    layers, counts = enumerate_words()
    ls = np.arange(4, LMAX + 1, 2)
    cov = covering(layers, ls)
    checks(layers, counts, cov, ls)
    b = []
    Lall = np.arange(LMAX + 1)
    # ---- (a) number of distinct rotations
    pa = N.Plot((0, 38), (0, 7), 4.2, 3.6, origin=(0.9, 0.7))
    b += pa.axes(xlabel="{\\scriptsize $L$}", ylabel="{\\scriptsize count}", xticks=[(10, "$10$"), (20, "$20$"), (30, "$30$")],
                 yticks=[(k, "$10^{%d}$" % k) for k in (1, 3, 5, 7)])
    b.append(pa.curve(Lall[:23], np.log10(2.0 ** (Lall[:23] + 1) - 1), "line width=0.9pt, ndInk!60, dash pattern=on 3pt off 2pt"))
    b.append(pa.curve(Lall, np.log10(counts), "line width=1.3pt, ndBlue"))
    b.append("\\node[note, anchor=east] at %s {words,\\\\ $2^{L+1}-1$};" % pa.pt(15.5, 6.2))
    b.append("\\node[note, anchor=north west, text=ndBlue] at %s {distinct\\\\ rotations $N_L$};" % pa.pt(22, 3.9))
    b.append("\\node[font=\\small, anchor=north] at %s {(a) products of $H$ and $T$};" % pa.pt(19, -1.25))

    # ---- (b) covering
    pb = N.Plot((0, 38), (-2.0, 0.6), 4.2, 3.6, origin=(6.6, 0.7))
    b += pb.axes(xlabel="{\\scriptsize $L$}", ylabel="{\\scriptsize angle}", xticks=[(10, "$10$"), (20, "$20$"), (30, "$30$")],
                 yticks=[(-2, "$10^{-2}$"), (-1, "$10^{-1}$"), (0, "$1$")])
    b.append(pb.curve(ls, np.log10(cov[:, 0]), "line width=1.2pt, ndRed"))
    b.append(pb.curve(ls, np.log10(cov[:, 1]), "line width=1.2pt, ndBlue"))
    b.append(pb.curve(Lall[3:], np.log10(bound_eps(counts[3:])), "line width=0.9pt, ndGold!85!ndInk, dash pattern=on 3pt off 2pt"))
    b.append("\\node[note, anchor=south west, text=ndRed] at %s {largest};" % pb.pt(25, np.log10(cov[list(ls).index(26), 0]) + 0.05))
    b.append("\\node[note, anchor=north east, text=ndBlue] at %s {mean};" % pb.pt(14, np.log10(cov[list(ls).index(14), 1]) - 0.1))
    b.append("\\node[note, anchor=west, text=ndGold!55!ndInk] (bd) at %s {counting bound $(6\\pi/N_L)^{1/3}$};" % pb.pt(3, -1.55))
    b.append("\\draw[pointer] (bd.north) to[bend left=15] %s;" % pb.pt(17, np.log10(bound_eps(counts[17])) - 0.04))
    b.append("\\node[font=\\small, anchor=north] at %s {(b) distance to the nearest};" % pb.pt(19, -2.0 - 0.64 * 2.6 / 3.6))

    # ---- (c) counting argument for n qubits
    ns = np.arange(1, 13)
    pc = N.Plot((0, 13), (0, 9), 4.2, 3.6, origin=(12.3, 0.7))
    b += pc.axes(xlabel="{\\scriptsize $n$}", ylabel="{\\scriptsize gates}", xticks=[(1, "$1$"), (4, "$4$"), (8, "$8$"), (12, "$12$")],
                 yticks=[(k, "$10^{%d}$" % k) for k in (1, 3, 5, 7, 9)])
    for eps, col, lab, yo in [(1e-2, "ndBlue", "$\\varepsilon=10^{-2}$", 0.3), (1e-6, "ndRed", "$\\varepsilon=10^{-6}$", 0.25)]:
        vals = np.log10([lower_bound_gates(n, eps) for n in ns])
        keep = vals <= 9.0
        b.append(pc.curve(ns[keep], vals[keep], "line width=1.2pt, %s" % col))
        for n, v in zip(ns[keep], vals[keep]):
            b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=2.2pt] at %s {};" % (col, pc.pt(n, v)))
    b.append("\\node[note, anchor=south east, text=ndRed] at %s {$\\varepsilon=10^{-6}$};" % pc.pt(5.4, np.log10(lower_bound_gates(5, 1e-6)) + 0.45))
    b.append("\\node[note, anchor=north west, text=ndBlue] at %s {$\\varepsilon=10^{-2}$};" % pc.pt(9.4, np.log10(lower_bound_gates(9, 1e-2)) - 0.45))
    b.append("\\node[note, anchor=north west] at %s {grows like $4^n$};" % pc.pt(1.0, 8.6))
    b.append("\\node[font=\\small, anchor=north] at %s {(c) most unitaries need at least};" % pc.pt(6.5, -0.64 * 9 / 3.6))
    N.write("ch10_counting", b)
    return counts, cov, ls


if __name__ == "__main__":
    counts, cov, ls = main()
    for L, (mx, mn) in zip(ls, cov):
        print("L=%2d  N=%7d  largest %.4f  mean %.4f  bound %.4f" % (L, counts[L], mx, mn, bound_eps(counts[L])))
    for n in (1, 2, 5, 10, 20):
        print("n=%d: L >= %.3g (eps 1e-2), %.3g (eps 1e-6)" % (n, lower_bound_gates(n, 1e-2), lower_bound_gates(n, 1e-6)))
    print("written figures/ch10_counting.tex")
