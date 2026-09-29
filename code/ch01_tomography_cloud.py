"""ch01_tomography_cloud.py -- qubit state tomography by linear inversion:
estimates of the Bloch vector from N shots per Pauli axis, for many repetitions,
shown in the (x,z) plane of the Bloch ball; some estimates are unphysical (outside
the ball) and are projected back.  Writes figures/ch01_tomography_cloud.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

rng = np.random.RandomState(5)
r = np.array([0.55, 0.0, 0.80])            # true Bloch vector, |r| = 0.97 (nearly pure)
def estimates(N, reps):
    out = []
    for _ in range(reps):
        est = np.array([2 * rng.binomial(N, (1 + ri) / 2) / N - 1 for ri in r])
        out.append(est)
    return np.array(out)
body = []
for k, (N, dx) in enumerate([(30, 0.0), (1000, 5.2)]):
    E = estimates(N, 150)
    outside = np.linalg.norm(E, axis=1) > 1
    body.append(r"\begin{scope}[shift={(%.2f,0)}, x=1.9cm, y=1.9cm]" % dx)
    body.append(r"\draw[thick] (0,0) circle (1); \draw[->, gray] (-1.25,0) -- (1.25,0) node[right, font=\scriptsize] {$r_x$}; \draw[->, gray] (0,-1.25) -- (0,1.25) node[above, font=\scriptsize] {$r_z$};")
    for e, o in zip(E, outside):
        body.append(r"\fill[%s, opacity=0.6] (%.3f,%.3f) circle (0.9pt);" % ("qocD" if o else "qocA", e[0], e[2]))
    body.append(r"\fill[qocC] (%.3f,%.3f) circle (2pt) node[below left, font=\scriptsize, text=qocC] {true $\rho$};" % (r[0], r[2]))
    if outside.any():
        e = E[outside][0]; p = e / np.linalg.norm(e)
        body.append(r"\draw[->, qocD, thick] (%.3f,%.3f) -- (%.3f,%.3f);" % (e[0], e[2], p[0], p[2]))
    body.append(r"\node[font=\small, align=center] at (0,-1.55) {$N=%d$ shots per axis\\ %d of %d estimates unphysical};" % (N, outside.sum(), len(E)))
    body.append(r"\end{scope}")
T.write_tikz("ch01_tomography_cloud", "\n".join(body))
print("written figures/ch01_tomography_cloud.tex")
