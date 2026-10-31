"""ch06_qfi.py -- the best measurement of a rotating qubit.

A qubit whose Bloch vector r turns in the equatorial plane at the angular
frequency omega.  After the time t its derivative with respect to omega is
dr = t (z x r), perpendicular to r and of length t |r|.  A projective
measurement along the unit vector m gives +1 with probability (1 + m.r)/2,
and its classical Fisher information about omega is
    F(m) = (m . dr)^2 / (1 - (m . r)^2).
The quantum Fisher information, the largest F over all measurements, is
    F_Q = |dr|^2 + (r . dr)^2 / (1 - |r|^2).
The figure draws, round each Bloch disc, the fraction F(m)/F_Q as a polar
curve:
  (a) a pure state, |r| = 1: every direction except that of r itself gives
      the whole of F_Q;
  (b) a mixed state, |r| = 0.6: only the directions perpendicular to r do.
Writes figures/ch06_qfi.tex.
"""
import numpy as np
import needham as N

T_EVOL = 1.0
PHI_R = np.radians(30.0)              # direction of the Bloch vector in the equatorial plane
ALPHA_M = np.radians(75.0)            # an example measurement direction


def fisher_m(alpha, r, dr):
    m = np.array([np.cos(alpha), np.sin(alpha)])
    return (m @ dr) ** 2 / (1.0 - (m @ r) ** 2)


def qfi(r, dr):
    nr = np.linalg.norm(r)
    extra = 0.0 if nr > 1 - 1e-12 else (r @ dr) ** 2 / (1 - nr ** 2)
    return dr @ dr + extra


def panel(b, c, length, title, note, R=2.0, s=1.45):
    """One Bloch disc with r, dr and the polar curve of F(m)/F_Q."""
    u = np.array([np.cos(PHI_R), np.sin(PHI_R)])
    v = np.array([-np.sin(PHI_R), np.cos(PHI_R)])
    r = length * u
    dr = T_EVOL * length * v                       # d r / d omega = t (z x r)
    FQ = qfi(r, dr)
    al = np.linspace(0, 2 * np.pi, 1441)
    eff = np.array([fisher_m(a, r, dr) if abs(np.sin(a - PHI_R)) > 1e-9 else np.nan for a in al]) / FQ
    # the largest classical Fisher information equals the quantum one
    assert abs(np.nanmax(eff) - 1.0) < 1e-6
    if length > 1 - 1e-12:
        ok = ~np.isnan(eff)
        assert np.allclose(eff[ok], 1.0)                 # pure: every direction but r gives F_Q
    else:
        assert abs(al[np.nanargmax(eff)] % np.pi - (PHI_R + np.pi / 2) % np.pi) < 0.01   # mixed: perpendicular to r
    P = lambda p: (c[0] + p[0], c[1] + p[1])
    b.append("\\fill[ndSky, fill opacity=0.55] (%.3f,%.3f) circle (%.3f);" % (*c, R))
    b.append("\\draw[ink] (%.3f,%.3f) circle (%.3f);" % (*c, R))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[right, font=\\scriptsize, text=ndInk] {$x$};"
             % (c[0] - R - 0.25, c[1], c[0] + R + 0.3, c[1]))
    b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[above, font=\\scriptsize, text=ndInk] {$y$};"
             % (c[0], c[1] - R - 0.25, c[0], c[1] + R + 0.3))
    # the polar curve: distance from the centre is s F(m)/F_Q
    pts = [P(s * e * np.array([np.cos(a), np.sin(a)])) for a, e in zip(al, eff) if not np.isnan(e)]
    if length > 1 - 1e-12:
        b.append("\\draw[line width=1.1pt, ndRed, fill=ndRose, fill opacity=0.5] (%.3f,%.3f) circle (%.3f);" % (*c, s))
        for a in (PHI_R, PHI_R + np.pi):                 # the one excluded direction
            b.append("\\filldraw[fill=white, draw=ndRed, line width=0.7pt] (%.3f,%.3f) circle (2.2pt);"
                     % P(s * np.array([np.cos(a), np.sin(a)])))
    else:
        b.append(N.polyline(pts, "line width=1.1pt, ndRed, fill=ndRose, fill opacity=0.5") + "")
    # an example measurement axis m
    mm = np.array([np.cos(ALPHA_M), np.sin(ALPHA_M)])
    b.append("\\draw[dashedink] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P(-R * mm), *P(R * mm)))
    b.append("\\draw[vec, ndGreen] (%.3f,%.3f) -- (%.3f,%.3f) node[pos=1.0, above left, font=\\small, text=ndGreen] {$\\vec m$};"
             % (*c, *P(0.75 * R * mm)))
    # the Bloch vector and its derivative
    b.append("\\draw[vec, ndInk] (%.3f,%.3f) -- (%.3f,%.3f);" % (*c, *P(R * r)))
    b.append("\\node[font=\\small, anchor=north west] at (%.3f,%.3f) {$\\Bloch$};" % P(R * r + np.array([0.02, -0.02])))
    b.append("\\draw[vec, ndBlue] (%.3f,%.3f) -- (%.3f,%.3f) node[pos=1.0, above right, font=\\small, text=ndBlue] {$\\partial_\\omega\\Bloch$};"
             % (*P(R * r), *P(R * r + 0.55 * R * dr / max(np.linalg.norm(dr), 1e-12) * np.linalg.norm(dr))))
    b.append("\\node[note, anchor=north] at (%.3f,%.3f) {%s};" % (c[0], c[1] - R - 0.45, note))
    b.append("\\node[font=\\small] at (%.3f,%.3f) {%s};" % (c[0], c[1] - R - 1.55, title))
    return FQ


def main():
    b = []
    fq_pure = panel(b, (0.0, 0.0), 1.0, "(a) a pure state, $\\abs{\\Bloch}=1$",
                    "every direction but that of $\\Bloch$\\\\ gives the whole of $\\mathcal F_Q$")
    fq_mixed = panel(b, (8.6, 0.0), 0.6, "(b) a mixed state, $\\abs{\\Bloch}=0.6$",
                     "only the directions perpendicular\\\\ to $\\Bloch$ give the whole of $\\mathcal F_Q$")
    assert abs(fq_pure - T_EVOL ** 2) < 1e-12 and abs(fq_mixed - (0.6 * T_EVOL) ** 2) < 1e-12
    b.append("\\node[note, anchor=north] (k) at (4.3,2.2) {red: the fraction\\\\ $\\mathcal F(\\vec m)/\\mathcal F_Q$ of the\\\\ quantum Fisher\\\\ information gained\\\\ by measuring along $\\vec m$,\\\\ drawn as the distance\\\\ from the centre};")
    b.append("\\node[note, anchor=north] at (4.3,-0.75) {blue: the change\\\\ of $\\Bloch$ with $\\omega$};")
    N.write("ch06_qfi", b)
    print("written figures/ch06_qfi.tex; F_Q pure %.3f, mixed %.3f" % (fq_pure, fq_mixed))


if __name__ == "__main__":
    main()
