"""ch01_measurement.py -- what an experiment returns: a qubit state on the Bloch
sphere, the Born probabilities of a Z measurement, and the histogram of a finite
number of shots.  Writes figures/ch01_measurement.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

rng = np.random.RandomState(1)
theta, phi = 0.7 * np.pi, 0.4 * np.pi           # a pure state |psi> = cos(th/2)|0> + e^{i phi} sin(th/2)|1>
r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
p0 = (1 + r[2]) / 2
shots = [10, 100, 1000]
freq = [rng.binomial(N, p0) / N for N in shots]

sphere = T.bloch_sphere(trajectories=[], radius=1.6, points=[(tuple(r), r"$\ket\psi$")])
sphere += "\n\\draw[very thick, qocA, ->] (0,0) -- (%.3f,%.3f);" % (1.6 * (0.9 * r[0] - 0.45 * r[1]), 1.6 * (r[2] - 0.2 * r[1]))
bars = r"""
\begin{scope}[shift={(4.2,-1.8)}]
\begin{axis}[width=5.2cm, height=4.2cm, ybar, bar width=9pt, ymin=0, ymax=1, ylabel={$\Prob(0)$}, xtick=data,
  symbolic x coords={Born, $N{=}10$, $N{=}100$, $N{=}1000$}, tick label style={font=\scriptsize}, label style={font=\small},
  nodes near coords, nodes near coords style={font=\tiny}, enlarge x limits=0.18, xticklabel style={rotate=25, anchor=north east}]
\addplot[fill=qocA!60, draw=qocA] coordinates {(Born,%.3f) ($N{=}10$,%.3f) ($N{=}100$,%.3f) ($N{=}1000$,%.3f)};
\end{axis}
\end{scope}
\node[font=\small, align=center] at (6.6,2.1) {measure $\sigma_z$: outcome $0$ w.p. $\tfrac{1+r_z}{2}$,\\ then the state is destroyed};
""" % (p0, freq[0], freq[1], freq[2])
T.write_tikz("ch01_measurement", sphere + bars)
print("written figures/ch01_measurement.tex  p0=%.3f freqs=%s" % (p0, freq))
