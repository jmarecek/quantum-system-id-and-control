"""ch02_tree.py -- measurements disturb: a probability tree and simulated shots.

Start in |0>.  Measuring sigma_z gives +1 with certainty, again and again.
Measuring sigma_x first sends the state to |+> or |-> with probability 1/2,
and a subsequent sigma_z measurement is then a fair coin.  The Born-rule
probabilities are computed from the amplitudes, and 1000 runs of each
sequence are simulated.  Writes figures/ch02_tree.tex.
"""
import numpy as np
import needham as N

K0, K1 = np.array([1, 0], complex), np.array([0, 1], complex)
KP, KM = (K0 + K1) / np.sqrt(2), (K0 - K1) / np.sqrt(2)
Z_BASIS = [(K0, "+1", "\\ket0"), (K1, "-1", "\\ket1")]
X_BASIS = [(KP, "+1", "\\ket+"), (KM, "-1", "\\ket-")]
SHOTS = 1000
rng = np.random.default_rng(7)


def measure(psi, basis):
    """Born rule: sample an outcome, return (index, post-measurement state)."""
    p = np.array([abs(np.vdot(b, psi)) ** 2 for b, _, _ in basis])
    j = rng.choice(len(basis), p=p / p.sum())
    return j, basis[j][0]


def simulate(sequence):
    counts = np.zeros(2, int)
    for _ in range(SHOTS):
        psi = K0
        for basis in sequence:
            j, psi = measure(psi, basis)
        counts[j] += 1                              # outcome of the last measurement
    return counts


def frac(p):
    """1 -> '1', 0.5 -> '\\tfrac12' (the only probabilities in this figure)."""
    return "1" if np.isclose(p, 1) else "\\tfrac12" if np.isclose(p, 0.5) else "%.2g" % p


def node(b, xy, label, col="ndInk"):
    b.append("\\node[draw=%s, line width=0.6pt, rounded corners=2pt, fill=white, inner sep=2pt, font=\\small] at (%.2f,%.2f) {$%s$};"
             % (col, xy[0], xy[1], label))


def edge(b, a, c, prob, col="ndInk!70"):
    b.append("\\draw[->, >=stealth, line width=0.7pt, %s, shorten <=9pt, shorten >=9pt] (%.2f,%.2f) -- (%.2f,%.2f)"
             " node[midway, sloped, above, font=\\scriptsize, text=ndInk] {%s};" % (col, a[0], a[1], c[0], c[1], prob))


def histogram(b, x0, y0, counts, title, col):
    W, H = 0.7, 1.8
    b.append("\\draw[faint] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0 - 0.2, y0, x0 + 2 * W + 0.5, y0))
    for k, c in enumerate(counts):
        h = H * c / SHOTS
        xa = x0 + k * (W + 0.3)
        b.append("\\fill[%s, fill opacity=0.75] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (col, xa, y0, xa + W, y0 + h))
        b.append("\\draw[ink, line width=0.5pt] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (xa, y0, xa + W, y0 + h))
        b.append("\\node[font=\\scriptsize, above] at (%.2f,%.2f) {%d};" % (xa + W / 2, y0 + h, c))
        b.append("\\node[font=\\scriptsize, below] at (%.2f,%.2f) {$%s$};" % (xa + W / 2, y0, "+1" if k == 0 else "-1"))
    b.append("\\node[note, anchor=south] at (%.2f,%.2f) {%s};" % (x0 + W + 0.15, y0 + H + 0.35, title))


def main():
    b = []
    # --- top: sigma_z twice
    y = 3.6
    node(b, (0, y), "\\ket0"); node(b, (3.2, y), "\\ket0"); node(b, (6.4, y), "\\ket0")
    edge(b, (0, y), (3.2, y), "$\\sigma_z$: $+1$, prob.\\ $1$"); edge(b, (3.2, y), (6.4, y), "$\\sigma_z$: $+1$, prob.\\ $1$")
    # --- bottom: sigma_x, then sigma_z; probabilities from the amplitudes
    y0 = 0.0
    node(b, (0, y0), "\\ket0")
    leaves = []
    for i, (bx, ox, lx) in enumerate(X_BASIS):
        px = abs(np.vdot(bx, K0)) ** 2
        yx = y0 + (1.0 if i == 0 else -1.0) * 1.1
        node(b, (3.2, yx), lx, "ndBlue")
        edge(b, (0, y0), (3.2, yx), "$\\sigma_x$: $%s$, $%s$" % (ox, frac(px)), "ndBlue!70")
        for j, (bz, oz, lz) in enumerate(Z_BASIS):
            pz = abs(np.vdot(bz, bx)) ** 2
            yz = yx + (0.55 if j == 0 else -0.55)
            node(b, (6.4, yz), lz, "ndRed")
            edge(b, (3.2, yx), (6.4, yz), "$\\sigma_z$: $%s$, $%s$" % (oz, frac(pz)), "ndRed!70")
            leaves.append(px * pz)
    assert np.allclose(leaves, 0.25)
    b.append("\\node[note, anchor=west] at (6.9,%.2f) {four outcomes,\\\\ each with prob.\\ $\\tfrac12\\cdot\\tfrac12=\\tfrac14$};" % y0)
    b.append("\\node[note, anchor=west] at (6.9,%.2f) {always $+1$: looking\\\\ does not disturb $\\ket0$};" % y)
    # --- simulated shots
    c1 = simulate([Z_BASIS, Z_BASIS])
    c2 = simulate([X_BASIS, Z_BASIS])
    histogram(b, 10.4, 2.4, c1, "last $\\sigma_z$, after $\\sigma_z$", "ndGreen")
    histogram(b, 10.4, -1.9, c2, "last $\\sigma_z$, after $\\sigma_x$", "ndRed")
    b.append("\\node[note] at (11.35,-2.75) {%d simulated runs each} ;" % SHOTS)
    N.write("ch02_tree", b)
    print("written figures/ch02_tree.tex; counts", c1.tolist(), c2.tolist())


if __name__ == "__main__":
    main()
