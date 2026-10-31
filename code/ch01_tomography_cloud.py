"""ch01_tomography_cloud.py -- qubit state tomography by linear inversion:
estimates of the Bloch vector from N shots per Pauli axis, for many repetitions,
shown in the (x,z) plane of the Bloch ball; some estimates are unphysical (outside
the ball) and are projected back.  Writes figures/ch01_tomography_cloud.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

rng = np.random.RandomState(5)
r = np.array([0.55, 0.0, 0.80])            # true Bloch vector, |r| = 0.97 (nearly pure)
SC = 2.0                                   # cm per unit of the Bloch ball


def estimates(N, reps):
    out = []
    for _ in range(reps):
        est = np.array([2 * rng.binomial(N, (1 + ri) / 2) / N - 1 for ri in r])
        out.append(est)
    return np.array(out)


def inset(E, outside, P, e, p, w=0.085, rad=0.85):
    """A magnifying glass on the cloud: the window of radius w about r, drawn with radius rad (cm) at lower left."""
    c = np.array(P(-0.36, -0.42))
    M = rad / w
    Q = lambda x, z: c + M * np.array([x - r[0], z - r[2]])
    out = ["\\draw[faint] (%.3f,%.3f) circle (%.3f);" % (*P(r[0], r[2]), w * SC)]
    for s in (1, -1):                                     # the two tangents from the window to the glass
        d = c - np.array(P(r[0], r[2]))
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        a, q = np.array(P(r[0], r[2])) + s * n * w * SC, c + s * n * rad
        out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f);" % (*a, *q))
    out.append("\\fill[white] (%.3f,%.3f) circle (%.3f);" % (*c, rad))
    out.append("\\begin{scope}\\clip (%.3f,%.3f) circle (%.3f);" % (*c, rad))
    disc = [Q(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 4000)]
    out.append("\\fill[ndSky!45] %s -- cycle;" % " -- ".join(N.pt(q) for q in disc[::8]))
    arc = [q for q in disc if np.linalg.norm(q - c) < 1.3 * rad]
    out.append(N.polyline(arc, "ink"))
    for x, o in zip(E, outside):
        out.append("\\fill[%s, fill opacity=0.8] %s circle (1.1pt);" % ("ndRed" if o else "ndBlue", N.pt(Q(x[0], x[2]))))
    out.append("\\draw[->, >=stealth, line width=0.8pt, ndRed] %s -- %s;" % (N.pt(Q(e[0], e[2])), N.pt(Q(p[0], p[2]))))
    out.append("\\node[dot, fill=ndGold, draw=ndInk, line width=0.3pt, minimum size=4.6pt] at %s {};" % N.pt(Q(r[0], r[2])))
    out.append("\\end{scope}")
    out.append("\\draw[ink, line width=0.6pt] (%.3f,%.3f) circle (%.3f);" % (*c, rad))
    m = c + rad * np.array([np.cos(np.radians(215)), np.sin(np.radians(215))])
    out.append("\\node[note, anchor=east] (g) at (%.3f,%.3f) {magnified\\\\ %s times};"
               % (*P(-1.12, -0.62), {5: "five", 10: "ten"}[round(M / SC)]))
    out.append("\\draw[pointer] (g.east) to[bend left=10] (%.3f,%.3f);" % tuple(m))
    return out


def main():
    b = []
    counts = {}
    for k, (n_shots, dx) in enumerate([(30, 0.0), (1000, 6.2)]):
        E = estimates(n_shots, 150)
        outside = np.linalg.norm(E, axis=1) > 1
        counts[n_shots] = int(outside.sum())
        P = lambda x, z: (dx + SC * x, SC * z)
        b.append("\\fill[ndSky!45] (%.3f,%.3f) circle (%.3f);" % (*P(0, 0), SC))
        b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[right, font=\\scriptsize, text=ndInk] {$r_x$};"
                 % (*P(-1.2, 0), *P(1.25, 0)))
        b.append("\\draw[faint, ->, >=stealth] (%.3f,%.3f) -- (%.3f,%.3f) node[above, font=\\scriptsize, text=ndInk] {$r_z$};"
                 % (*P(0, -1.2), *P(0, 1.25)))
        b.append("\\draw[ink] (%.3f,%.3f) circle (%.3f);" % (*P(0, 0), SC))
        for e, o in zip(E, outside):
            b.append("\\fill[%s, fill opacity=0.75] (%.3f,%.3f) circle (0.9pt);" % ("ndRed" if o else "ndBlue", *P(e[0], e[2])))
        b.append("\\node[dot, fill=ndGold, draw=ndInk, line width=0.3pt, minimum size=4.6pt] at (%.3f,%.3f) {};" % P(r[0], r[2]))
        if outside.any():
            e = E[outside][0]
            p = e / np.linalg.norm(e)
            b.append("\\draw[->, >=stealth, line width=0.9pt, ndRed] (%.3f,%.3f) -- (%.3f,%.3f);" % (*P(e[0], e[2]), *P(p[0], p[2])))
        b.append("\\node[font=\\small, align=center] at (%.3f,%.3f) {$N=%d$ shots per axis\\\\ %d of %d estimates unphysical};"
                 % (*P(0, -1.5), n_shots, outside.sum(), len(E)))
        if k == 0:
            b.append("\\node[note, anchor=east, text=ndRed] (o) at (%.3f,%.3f) {outside the ball:\\\\ not a state};" % P(-0.35, 1.32))
            q = E[outside][np.argmax(E[outside][:, 2] - E[outside][:, 0])]
            b.append("\\draw[pointer] (o.east) to[bend left=20] (%.3f,%.3f);" % P(q[0], q[2]))
            b.append("\\node[note, anchor=east, text=ndGold!80!ndInk] (t) at (%.3f,%.3f) {true state,\\\\ $\\abs{\\Bloch}=%.2f$};"
                     % (*P(-0.06, 0.42), np.linalg.norm(r)))
            b.append("\\draw[pointer] (t.east) to[bend right=25] (%.3f,%.3f);" % P(r[0] - 0.04, r[2] - 0.02))
        else:
            b += inset(E, outside, P, e, p)
    assert counts == {30: 74, 1000: 9}                 # quoted in the figure
    N.write("ch01_tomography_cloud", b)
    print("written figures/ch01_tomography_cloud.tex; unphysical:", counts)


if __name__ == "__main__":
    main()
