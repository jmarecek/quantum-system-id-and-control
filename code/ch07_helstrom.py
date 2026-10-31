"""ch07_helstrom.py -- the best single-shot test between two qubit states.

Two mixed states rho and sigma of a qubit, with Bloch vectors r and s in the
(x, z) plane of the Bloch ball.  A projective measurement along the unit
vector m has the projector E = (1 + m.sigma)/2 for the outcome +1, and
    Tr E (rho - sigma) = m . (r - s) / 2 .
On the outcome +1 we guess rho, on -1 sigma; with equal priors the guess is
right with probability (1 + m.(r - s)/2)/2.  The best axis is
n = (r - s)/|r - s|, along the chord, where the shadows of the two states on
the axis are as far apart as the states themselves, |r - s| = 2D, and the
success probability is (1 + D)/2 (Helstrom).  Any other axis m sees the
shorter shadow |m.(r - s)|.  The figure draws the disc, the two states, the
best axis with its projector, and one worse axis.
Writes figures/ch07_helstrom.tex.
"""
import numpy as np
import needham as N

R_RHO = np.array([0.62, 0.38])          # (x, z) of rho
R_SIG = np.array([0.18, -0.62])         # (x, z) of sigma
ALPHA_M = np.radians(148.0)             # a worse measurement axis

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def state(v):
    """Density matrix with Bloch vector (v[0], 0, v[1])."""
    return 0.5 * (I2 + v[0] * SX + v[1] * SZ)


def projector(m):
    return 0.5 * (I2 + m[0] * SX + m[1] * SZ)


def main():
    rho, sig = state(R_RHO), state(R_SIG)
    d = R_RHO - R_SIG
    D = 0.5 * np.abs(np.linalg.eigvalsh(rho - sig)).sum()
    assert abs(D - 0.5 * np.linalg.norm(d)) < 1e-12                  # D = |r - s|/2 for a qubit
    n = d / np.linalg.norm(d)
    # Helstrom: the projector onto the positive part of rho - sigma is the measurement along n
    w, V = np.linalg.eigh(rho - sig)
    Ppos = V[:, w > 0] @ V[:, w > 0].conj().T
    assert np.allclose(Ppos, projector(n))
    assert abs(np.real(np.trace(Ppos @ (rho - sig))) - D) < 1e-12
    # scan of axes: the bias Tr E(rho - sigma) is largest, and equal to D, along n
    al = np.linspace(0, 2 * np.pi, 3601)
    bias = np.array([np.real(np.trace(projector((np.cos(a), np.sin(a))) @ (rho - sig))) for a in al])
    assert abs(bias.max() - D) < 1e-6 and abs(al[bias.argmax()] - np.arctan2(n[1], n[0]) % (2 * np.pi)) < 2e-3
    # no POVM element 0 <= E <= 1 does better
    rng = np.random.default_rng(1)
    for _ in range(2000):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        ev, U = np.linalg.eigh(A + A.conj().T)
        E = U @ np.diag(rng.random(2)) @ U.conj().T                  # random effect, eigenvalues in [0, 1]
        assert np.real(np.trace(E @ (rho - sig))) <= D + 1e-12
    m = np.array([np.cos(ALPHA_M), np.sin(ALPHA_M)])
    gap_m = abs(m @ d)
    assert gap_m < np.linalg.norm(d)
    P_succ = 0.5 * (1 + D)
    print("r = %s, s = %s: |r - s| = %.3f, D = %.3f, success (1 + D)/2 = %.3f; worse axis at %.0f deg: shadows %.3f apart, success %.3f"
          % (R_RHO, R_SIG, np.linalg.norm(d), D, P_succ, np.degrees(ALPHA_M), gap_m, 0.5 * (1 + gap_m / 2)))

    R = 2.2                                                           # cm per unit of the Bloch vector
    P = lambda v: R * np.asarray(v)
    b = []
    b.append("\\fill[ndSky, fill opacity=0.55] (0,0) circle (%.3f);" % R)
    b.append("\\draw[ink] (0,0) circle (%.3f);" % R)
    b.append("\\draw[faint, ->, >=stealth] (%.3f,0) -- (%.3f,0) node[right, font=\\scriptsize, text=ndInk] {$x$};" % (-R - 0.2, R + 0.3))
    b.append("\\draw[faint, ->, >=stealth] (0,%.3f) -- (0,%.3f) node[above, font=\\scriptsize, text=ndInk] {$z$};" % (-R - 0.2, R + 0.3))
    # the worse axis m, faint, with its shadows (closer together)
    b.append("\\draw[ndGreen!70, line width=0.6pt] %s -- %s;" % (N.pt(P(-m)), N.pt(P(m))))
    b.append("\\node[font=\\small, text=ndGreen, anchor=south] at %s {$\\vec m$};" % N.pt(P(1.03 * m)))
    fm = [(v @ m) * m for v in (R_RHO, R_SIG)]
    b.append("\\draw[line width=2.4pt, ndGreen!55] %s -- %s;" % (N.pt(P(fm[0])), N.pt(P(fm[1]))))
    for f, col in zip(fm, ("ndBlue", "ndRed")):
        b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=3.0pt] at %s {};" % (col, N.pt(P(f))))
    # the best axis n, through the centre, and its projector
    b.append("\\draw[ink, line width=0.9pt] %s -- %s;" % (N.pt(P(-n)), N.pt(P(n))))
    b.append("\\node[dot] at %s {};" % N.pt(P(n)))
    b.append("\\node[dot, fill=white, draw=ndInk] at %s {};" % N.pt(P(-n)))
    b.append("\\node[font=\\small, anchor=south west] at %s {$\\vec n$};" % N.pt(P(n) + np.array([0.03, 0.0])))
    b.append("\\node[note, anchor=north east] at %s {outcome $-1$:\\\\ guess $\\sigma$};" % N.pt(P(-n) + np.array([-0.05, -0.02])))
    # shadows of the two states on the best axis
    feet = []
    for v, col in ((R_RHO, "ndBlue"), (R_SIG, "ndRed")):
        foot = (v @ n) * n
        feet.append(foot)
        b.append("\\draw[%s, line width=0.5pt, densely dashed] %s -- %s;" % (col, N.pt(P(v)), N.pt(P(foot))))
    assert abs(np.linalg.norm(feet[0] - feet[1]) - np.linalg.norm(d)) < 1e-12   # shadows as far apart as the states
    b.append("\\draw[line width=2.4pt, ndGold!80] %s -- %s;" % (N.pt(P(feet[1])), N.pt(P(feet[0]))))
    for f, col in zip(feet, ("ndBlue", "ndRed")):
        b.append("\\node[circle, fill=%s, inner sep=0pt, minimum size=3.4pt] at %s {};" % (col, N.pt(P(f))))
    left = np.array([-n[1], n[0]])                                     # normal to n, away from the chord
    b.append("\\node[font=\\small, text=ndGold!60!ndInk, anchor=east] at %s {$2D$};" % N.pt(P(-0.3 * n) + 0.12 * left))
    # the chord and the two Bloch vectors
    b.append("\\draw[ndInk!70, line width=0.7pt] %s -- %s;" % (N.pt(P(R_SIG)), N.pt(P(R_RHO))))
    b.append("\\draw[vec, ndBlue] (0,0) -- %s;" % N.pt(P(R_RHO)))
    b.append("\\draw[vec, ndRed] (0,0) -- %s;" % N.pt(P(R_SIG)))
    b.append("\\node[font=\\small, text=ndBlue, anchor=west] at %s {$\\Bloch_\\rho$};" % N.pt(P(R_RHO) + np.array([0.08, 0.0])))
    b.append("\\node[font=\\small, text=ndRed, anchor=west] at %s {$\\Bloch_\\sigma$};" % N.pt(P(R_SIG) + np.array([0.08, -0.02])))
    # notes
    b.append("\\node[note, anchor=west] (best) at (%.3f,1.45) {the best axis $\\vec n$ is parallel to\\\\ the chord, with the projector\\\\ $E=\\tfrac12(\\Id+\\vec n\\cdot\\vec\\sigma)$; the shadows\\\\ of the states on it are as far\\\\ apart as the states, $2D$};" % (R + 0.75))
    b.append("\\draw[pointer] (best.west) to[bend right=15] %s;" % N.pt(P(0.93 * n)))
    b.append("\\node[note, anchor=west] at (%.3f,0.0) {guess $\\rho$ on outcome $+1$:\\\\ right with probability\\\\ $\\tfrac12(1+D)=%.2f$};" % (R + 0.75, P_succ))
    b.append("\\node[note, anchor=west, text=ndGreen!80!ndInk] (worse) at (%.3f,-1.45) {any other axis $\\vec m$ casts\\\\ closer shadows, and tells\\\\ the states apart less well};" % (R + 0.75))
    mr = m if m[0] > 0 else -m                                       # the end of m nearer the notes
    b.append("\\draw[pointer] (worse.west) to[bend left=12] %s;" % N.pt(P(0.88 * mr)))
    N.write("ch07_helstrom", b)
    print("written figures/ch07_helstrom.tex")


if __name__ == "__main__":
    main()
