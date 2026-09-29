"""ch02_born.py -- the Born rule as the squared length of a shadow.

A real slice of a qubit's Hilbert space: the unit vector |psi> at angle theta
from |0>, its orthogonal projections onto the eigenvectors |0>, |1> of the
measured observable, and the collapse onto |0>.  Writes figures/ch02_born.tex.
"""
import numpy as np
import needham as N

THETA = np.radians(34.0)
R = 3.0                                   # length of a unit vector on the page


def main():
    psi = R * np.array([np.cos(THETA), np.sin(THETA)])
    p0, p1 = np.cos(THETA) ** 2, np.sin(THETA) ** 2
    assert abs(p0 + p1 - 1) < 1e-12
    b = []
    # unit circle (quarter) and the axes spanned by the eigenvectors
    t = np.linspace(0, np.pi / 2, 60)
    b.append(N.polyline([(R * np.cos(s), R * np.sin(s)) for s in t], "faint"))
    b.append("\\draw[faint, ->, >=stealth] (-0.3,0) -- (%.2f,0);" % (R + 0.9))
    b.append("\\draw[faint, ->, >=stealth] (0,-0.3) -- (0,%.2f);" % (R + 0.7))
    # the two shadows, drawn as thick coloured segments on the axes
    # (drawn just beside the axes, like shadows cast by the state)
    OFF = 0.16
    b.append("\\draw[line width=3.4pt, ndGold] (0,%.3f) -- (%.3f,%.3f);" % (-OFF, psi[0], -OFF))
    b.append("\\draw[line width=3.4pt, ndGreen] (%.3f,0) -- (%.3f,%.3f);" % (-OFF, -OFF, psi[1]))
    b.append("\\draw[dashedink] %s -- (%.3f,%.3f);" % (N.pt(psi), psi[0], -OFF))
    b.append("\\draw[dashedink] %s -- (%.3f,%.3f);" % (N.pt(psi), -OFF, psi[1]))
    # basis vectors and the state
    b.append("\\draw[vec, ndInk] (0,0) -- (%.3f,0) node[below right] {$\\ket0$};" % R)
    b.append("\\draw[vec, ndInk] (0,0) -- (0,%.3f) node[above left] {$\\ket1$};" % R)
    b.append("\\draw[vec, ndBlue] (0,0) -- %s node[above right] {$\\ket\\psi$};" % N.pt(psi))
    # angle
    ta = np.linspace(0, THETA, 20)
    b.append(N.polyline([(0.8 * np.cos(s), 0.8 * np.sin(s)) for s in ta], "ink"))
    b.append("\\node at (%.3f,%.3f) {$\\theta$};" % (1.05 * np.cos(THETA / 2), 1.05 * np.sin(THETA / 2)))
    # labels of the shadows
    b.append("\\node[ndGold!70!black, below] at (%.3f,-0.25) {$\\braket{0|\\psi}=\\cos\\theta$};" % (psi[0] / 2))
    b.append("\\node[ndGreen!80!black, left] at (-0.25,%.3f) {$\\braket{1|\\psi}=\\sin\\theta$};" % (psi[1] / 2))
    # collapse: psi is replaced by the normalised shadow |0>
    tc = np.linspace(THETA - 0.06, 0.1, 30)
    arc = [(1.12 * R * np.cos(s), 1.12 * R * np.sin(s)) for s in tc]
    b.append(N.polyline(arc, "->, >=stealth, line width=0.8pt, ndRed"))
    b.append("\\node[note, right] at (%.3f,%.3f) {outcome $0$:\\\\ \\emph{collapse} to $\\ket0$};"
             % (1.14 * R * np.cos(THETA / 2) + 0.05, 1.14 * R * np.sin(THETA / 2)))
    # the message
    b.append("\\node[note, anchor=south west] (m) at (0.6,%.2f) "
             "{probability of outcome $j$ $=$ (length of the shadow on $\\ket j$)$^2$,\\\\"
             "and the shadows obey Pythagoras: $\\cos^2\\theta+\\sin^2\\theta=1$};" % (R + 0.35))
    b.append("\\draw[pointer] (m.west) to[bend right=35] (%.3f,%.3f);" % (-OFF - 0.08, 0.8 * psi[1]))
    N.write("ch02_born", b)
    print("written figures/ch02_born.tex; P(0)=%.3f, P(1)=%.3f" % (p0, p1))


if __name__ == "__main__":
    main()
