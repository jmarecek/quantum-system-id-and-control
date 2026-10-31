"""ch07_bures.py -- the Bures geometry of a qubit is that of a hemisphere.

For qubit states with Bloch vectors r and s the fidelity is
    F = (1/2) (1 + r.s + sqrt((1 - |r|^2)(1 - |s|^2))).
Lift each Bloch vector to the unit vector R = (r, sqrt(1 - |r|^2)) of R^4,
which lies on the upper hemisphere of the unit 3-sphere.  Then
F = (1/2)(1 + R.S) = cos^2(theta/2), theta the angle between R and S, so the
Bures angle arccos sqrt F is theta/2: the Bloch ball with the Bures metric is
a hemisphere of radius 1/2.  Pure states (|r| = 1) lie on its rim, the
maximally mixed state at its pole.  The line element is
    ds_B^2 = (1/4) (|dr|^2 + (r.dr)^2 / (1 - |r|^2)).
  (a) the (x, z) disc of the Bloch ball and the hemisphere above it, two
      mixed states, the great circle between their lifts (the Bures geodesic),
      its shadow on the disc, which bows outwards, and the straight chord (the
      geodesic of the trace and Hilbert--Schmidt distances); the diameter of
      diagonal states lifts to a great semicircle, the classical picture;
  (b) indicatrices: at points of the disc, the displacements of Bures length
      eps form an ellipse with tangential semi-axis 2 eps and radial semi-axis
      2 eps sqrt(1 - |r|^2); the trace distance gives the circle of radius
      2 eps everywhere.
Writes figures/ch07_bures.tex.
"""
import numpy as np
from scipy.linalg import sqrtm
import needham as N

R_RHO = 0.80 * np.array([np.cos(np.radians(-35.0)), np.sin(np.radians(-35.0))])    # (x, z) of rho
R_SIG = 0.80 * np.array([np.cos(np.radians(125.0)), np.sin(np.radians(125.0))])    # (x, z) of sigma
EPS = 0.05                                     # Bures length of the drawn indicatrices

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def state(r):
    """Density matrix with Bloch vector r (three components)."""
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def fidelity(rho, sigma):
    s = sqrtm(rho)
    return np.real(np.trace(sqrtm(s @ sigma @ s))) ** 2


def fid_qubit(r, s):
    return 0.5 * (1 + r @ s + np.sqrt(max(0.0, (1 - r @ r) * (1 - s @ s))))


def lift(r):
    return np.append(r, np.sqrt(max(0.0, 1 - r @ r)))


def ds2(r, dr):
    """Bures line element (squared) at the Bloch vector r."""
    return 0.25 * (dr @ dr + (r @ dr) ** 2 / (1 - r @ r))


def checks():
    rng = np.random.default_rng(3)
    for _ in range(200):
        r, s = [v * rng.random() ** (1 / 3) / np.linalg.norm(v) for v in rng.normal(size=(2, 3))]
        F = fidelity(state(r), state(s))
        assert abs(F - fid_qubit(r, s)) < 1e-8                     # the qubit formula
        assert abs(F - 0.5 * (1 + lift(r) @ lift(s))) < 1e-8       # F = (1 + R.S)/2 = cos^2(theta/2)
        # the line element: Bures angle of a small step against ds_B
        if r @ r < 0.95:
            dr = 1e-5 * rng.normal(size=3)
            A = np.arccos(np.sqrt(min(1.0, fid_qubit(r, r + dr))))
            assert abs(A ** 2 / ds2(r, dr) - 1) < 2e-3
    # the indicatrix: radial step 2 eps sqrt(1 - r^2) and tangential step 2 eps both have Bures length eps
    r = np.array([0.7, 0.0, 0.3]); u = r / np.linalg.norm(r); t = np.array([0.0, 1.0, 0.0])
    e = 1e-4
    for step in (2 * e * np.sqrt(1 - r @ r) * u, 2 * e * t):
        A = np.arccos(np.sqrt(fid_qubit(r, r + step)))
        assert abs(A / e - 1) < 2e-3


def main():
    checks()
    r3, s3 = np.array([R_RHO[0], 0, R_RHO[1]]), np.array([R_SIG[0], 0, R_SIG[1]])
    Rl, Sl = lift(R_RHO), lift(R_SIG)
    theta = np.arccos(Rl @ Sl)
    F = fidelity(state(r3), state(s3))
    A = np.arccos(np.sqrt(F))
    assert abs(A - theta / 2) < 1e-8
    D = 0.5 * np.linalg.norm(R_RHO - R_SIG)
    dB = np.sqrt(2 - 2 * np.sqrt(F))
    # the great-circle arc between the lifts, and its shadow on the disc
    k = np.linspace(0, 1, 121)
    arc = np.array([(np.sin((1 - a) * theta) * Rl + np.sin(a * theta) * Sl) / np.sin(theta) for a in k])
    shadow = arc[:, :2]
    mid_chord = 0.5 * (R_RHO + R_SIG)
    assert np.linalg.norm(shadow[60]) > np.linalg.norm(mid_chord)             # the shadow bows outwards
    print("rho at %s, sigma at %s (|r| = %.2f, %.2f): F = %.3f, Bures angle %.3f = theta/2 with theta = %.3f; "
          "Bures distance %.3f; trace distance %.3f; HS distance %.3f"
          % (np.round(R_RHO, 3), np.round(R_SIG, 3), np.linalg.norm(R_RHO), np.linalg.norm(R_SIG), F, A, theta, dB, D, 2 * D / np.sqrt(2)))

    b = []
    # ---- (a) the hemisphere over the disc; page coordinates (x, z, w) with the lift w vertical
    V = N.View(az=45.0, el=38.0, R=2.25, shift=(0.0, 0.0))
    for p in (Rl, Sl):
        assert p @ V.d > 0                                                    # both lifts on the visible side
    tt = np.linspace(0, 2 * np.pi, 361)
    rim = np.stack([np.cos(tt), np.sin(tt), 0 * tt], 1)
    b.append("\\fill[ndSky, fill opacity=0.55] %s -- cycle;" % " -- ".join(N.pt(V.P(p)) for p in rim))
    # the dome: the sphere's outline above the rim, closed along the front of the rim
    u = np.cross(V.d, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(V.d, u)
    if v[2] < 0:
        v = -v
    sil = np.array([np.cos(a) * u + np.sin(a) * v for a in np.linspace(0, np.pi, 181)])
    assert np.all(sil[:, 2] >= -1e-9)
    a_end = np.arctan2(sil[-1, 1], sil[-1, 0]); a_start = np.arctan2(sil[0, 1], sil[0, 0])
    steps = np.linspace(0, (a_start - a_end) % (2 * np.pi), 120)
    walk = np.stack([np.cos(a_end + steps), np.sin(a_end + steps), 0 * steps], 1)
    if np.mean(walk @ V.d) < 0:                                               # go round the other way
        steps = steps - 2 * np.pi * steps / steps[-1]
        walk = np.stack([np.cos(a_end + steps), np.sin(a_end + steps), 0 * steps], 1)
    assert np.all(walk @ V.d >= -1e-6)
    b.append("\\fill[ndSky!60!white, fill opacity=0.35] %s -- cycle;" % " -- ".join(N.pt(V.P(p)) for p in list(sil) + list(walk)))
    for lat in (30, 60):                                                      # faint parallels and meridians
        c_, s_ = np.cos(np.radians(lat)), np.sin(np.radians(lat))
        b += V.curve(np.stack([c_ * np.cos(tt), c_ * np.sin(tt), s_ + 0 * tt], 1), "line width=0.3pt, ndInk!22")
    ph = np.linspace(0, np.pi, 181)
    for lon in (0, 45, 135):
        lo = np.radians(lon)
        b += V.curve(np.stack([np.cos(lo) * np.cos(ph), np.sin(lo) * np.cos(ph), np.sin(ph)], 1), "line width=0.3pt, ndInk!22")
    b += V.curve(rim, "ink", "dashedink")
    b.append(N.polyline([V.P(p) for p in sil], "ink"))
    # the x- and z-diameters of the disc, labelled at the rim
    b.append("\\draw[faint] %s -- %s;" % (V.pt((-1, 0, 0)), V.pt((1, 0, 0))))
    b.append("\\draw[faint] %s -- %s;" % (V.pt((0, -1, 0)), V.pt((0, 1, 0))))
    b.append("\\node[font=\\scriptsize, anchor=north east] at %s {$x$};" % N.pt(V.P((1, 0, 0)) + np.array([-0.02, -0.02])))
    b.append("\\node[font=\\scriptsize, anchor=north west] at %s {$z$};" % N.pt(V.P((0, 1, 0)) + np.array([0.02, -0.02])))
    # the diagonal states: the z-diameter, lifted to a great semicircle
    zz = np.linspace(-1, 1, 161)
    b += V.curve(np.stack([0 * zz, zz, np.sqrt(1 - zz ** 2)], 1), "line width=0.8pt, ndGold", "line width=0.5pt, ndGold, densely dotted")
    # the chord, the Bures geodesic and its shadow
    b.append("\\draw[ink, line width=1.0pt] %s -- %s;" % (V.pt((*R_RHO, 0)), V.pt((*R_SIG, 0))))
    b.append(N.polyline([V.P((p[0], p[1], 0)) for p in shadow], "ndBlue, line width=0.8pt, dash pattern=on 3pt off 1.5pt"))
    b += V.curve(arc, "ndBlue, line width=1.5pt", "ndBlue, line width=0.8pt, densely dotted")
    for p2, P4, col, lab, off in [(R_RHO, Rl, "ndBlue", "\\rho", (-0.2, 0.04)), (R_SIG, Sl, "ndRed", "\\sigma", (0.2, 0.06))]:
        b.append("\\draw[faint, densely dashed] %s -- %s;" % (V.pt((*p2, 0)), V.pt(P4)))
        b.append("\\node[dot, fill=%s] at %s {};" % (col, V.pt((*p2, 0))))
        b.append("\\node[dot, fill=%s] at %s {};" % (col, V.pt(P4)))
        b.append("\\node[font=\\small, text=%s] at %s {$%s$};" % (col, N.pt(V.P((*p2, 0)) + np.array(off)), lab))
    b.append("\\node[dot] at %s {};" % V.pt((0, 0, 1)))
    b.append("\\node[font=\\small, anchor=south west, inner sep=1pt] at %s {$\\tfrac12\\Id$};" % N.pt(V.P((0, 0, 1)) + np.array([0.05, 0.02])))
    # notes for (a)
    top = V.P((0, 0, 1))
    b.append("\\node[note, anchor=south west] (lift) at %s {lift each state onto the\\\\ hemisphere: the Bures angle\\\\ is half the angle between\\\\ the lifts};" % N.pt(top + np.array([0.9, -0.05])))
    b.append("\\draw[pointer] (lift.south west) to[bend right=15] %s;" % N.pt(V.P(arc[85]) + np.array([0.03, 0.06])))
    q = np.array([np.cos(np.radians(118)), np.sin(np.radians(118)), 0.0])
    b.append("\\node[note, anchor=west] (pure) at %s {pure states:\\\\ the rim};" % N.pt(V.P(q) + np.array([0.55, -0.35])))
    b.append("\\draw[pointer] (pure.west) to[bend left=20] %s;" % V.pt(q))
    gz = np.array([0, -0.8, 0.6])
    b.append("\\node[note, anchor=east, text=ndGold!70!ndInk] (diag) at %s {diagonal states:\\\\ a great semicircle,\\\\ the classical picture};" % N.pt(V.P(gz) + np.array([-0.75, 0.55])))
    b.append("\\draw[pointer] (diag.east) to[bend left=15] %s;" % N.pt(V.P(gz) + np.array([-0.05, 0.0])))
    b.append("\\node[note, anchor=north] at %s {blue: the Bures geodesic and its shadow,\\\\ which bows outwards; black: the chord,\\\\ the geodesic of the trace distance};" % N.pt(V.P((0, 0, 0)) + np.array([0.0, -1.55])))
    b.append("\\node[font=\\small] at %s {(a) the Bloch disc lifted to a hemisphere};" % N.pt(V.P((0, 0, 0)) + np.array([0.0, -2.9])))

    # ---- (b) indicatrices on the Bloch disc
    C = np.array([8.6, 0.25]); Rb = 2.2
    Q = lambda p: C + Rb * np.asarray(p)
    b.append("\\fill[ndSky, fill opacity=0.55] %s circle (%.3f);" % (N.pt(C), Rb))
    b.append("\\draw[ink] %s circle (%.3f);" % (N.pt(C), Rb))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[right, font=\\scriptsize, text=ndInk] {$x$};" % (N.pt(C + [-Rb - 0.2, 0]), N.pt(C + [Rb + 0.3, 0])))
    b.append("\\draw[faint, ->, >=stealth] %s -- %s node[above, font=\\scriptsize, text=ndInk] {$z$};" % (N.pt(C + [0, -Rb - 0.2]), N.pt(C + [0, Rb + 0.3])))
    rings = [(0.0, 1, 0.0), (0.45, 6, 0.0), (0.75, 10, 18.0), (0.94, 16, 0.0)]
    ph = np.linspace(0, 2 * np.pi, 49)
    for rad, cnt, off in rings:
        for k_ in range(cnt):
            a = np.radians(off) + 2 * np.pi * k_ / cnt
            uu = np.array([np.cos(a), np.sin(a)]); tv = np.array([-uu[1], uu[0]])
            p = rad * uu
            rr, tr = 2 * EPS * np.sqrt(1 - rad ** 2), 2 * EPS
            # check the drawn semi-axes against the line element
            if rad > 0:
                assert abs(np.sqrt(ds2(np.array([p[0], 0, p[1]]), np.array([rr * uu[0], 0, rr * uu[1]]))) - EPS) < 1e-12
            ell = [Q(p + rr * np.cos(q) * uu + tr * np.sin(q) * tv) for q in ph]
            circ = [Q(p + tr * np.array([np.cos(q), np.sin(q)])) for q in ph]
            b.append(N.polyline(circ, "line width=0.3pt, ndInk!45, densely dashed"))
            b.append(N.polyline(ell, "line width=0.6pt, ndBlue, fill=ndBlue!25"))
    b.append("\\node[note, anchor=west] (nb) at %s {near the pure states the\\\\ Bures ellipses are thin:\\\\ a small step inwards\\\\ is a long way};" % N.pt(C + [Rb + 0.35, 1.0]))
    b.append("\\draw[pointer] (nb.west) to[bend right=20] %s;" % N.pt(Q(0.94 * np.array([np.cos(np.radians(22.5)), np.sin(np.radians(22.5))]) + np.array([0.04, 0.0]))))
    b.append("\\node[note, anchor=west] (nt) at %s {dashed: the trace\\\\ distance, the same\\\\ circle everywhere};" % N.pt(C + [Rb + 0.35, -1.0]))
    b.append("\\draw[pointer] (nt.west) to[bend left=15] %s;" % N.pt(Q(0.45 * np.array([np.cos(np.radians(-60)), np.sin(np.radians(-60))]) + np.array([2 * EPS, 0.0]))))
    b.append("\\node[font=\\small] at %s {(b) small balls of the Bures metric and of $D$};" % N.pt(np.array([C[0] + 0.9, -2.9])))
    N.write("ch07_bures", b)
    print("written figures/ch07_bures.tex")


if __name__ == "__main__":
    main()
