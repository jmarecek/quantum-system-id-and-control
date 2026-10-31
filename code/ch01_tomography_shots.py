"""ch01_tomography_shots.py -- error of qubit state tomography versus the number
of shots: linear inversion, its projection onto the Bloch ball, and the
1/sqrt(N) reference; also the same for the pure and a mixed state.
Writes figures/ch01_tomography_shots.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

rng = np.random.RandomState(11)


def run(r, Ns, reps=400):
    li, pr = [], []
    for N in Ns:
        E = np.array([[2 * rng.binomial(N, (1 + ri) / 2) / N - 1 for ri in r] for _ in range(reps)])
        err_li = np.linalg.norm(E - r, axis=1) / 2                      # trace distance = |r - r'|/2
        P = np.where((np.linalg.norm(E, axis=1) > 1)[:, None], E / np.linalg.norm(E, axis=1)[:, None], E)
        err_pr = np.linalg.norm(P - r, axis=1) / 2
        li.append(err_li.mean()); pr.append(err_pr.mean())
    return np.array(li), np.array(pr)


def main():
    Ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    li_p, pr_p = run(np.array([0.0, 0.0, 1.0]), Ns)      # pure state
    li_m, pr_m = run(np.array([0.3, 0.0, 0.4]), Ns)      # mixed state
    lx = np.log10(Ns)
    slope = lambda y: np.polyfit(lx, np.log10(y), 1)[0]
    assert all(abs(slope(y) + 0.5) < 0.03 for y in (li_p, pr_p, li_m, pr_m))       # the 1/sqrt(N) law
    assert pr_p[0] < 0.95 * li_p[0] and np.all(pr_p <= li_p)                          # projection helps a pure state ...
    assert np.all(pr_p[4:] > 0.99 * li_p[4:])                                         # ... but only for few shots
    assert np.all(np.abs(pr_m[2:] / li_m[2:] - 1) < 1e-3)                             # and is neutral for a mixed one

    b = []
    Pl = N.Plot((0.85, 4.15), (-2.7, -0.5), 9.0, 5.6)
    b += Pl.axes("shots per axis $N$", "mean trace distance to $\\rho$",
                 xticks=[(k, "$10^{%d}$" % k) for k in range(1, 5)],
                 yticks=[(-2, "$10^{-2}$"), (-1, "$10^{-1}$")])
    ref = lambda x: -0.5 * x
    b.append(Pl.curve(lx, ref(lx), "line width=0.6pt, ndInk!40"))
    for li, pr, col in [(li_p, pr_p, "ndBlue"), (li_m, pr_m, "ndGold")]:
        b.append(Pl.curve(lx, np.log10(li), "line width=1.0pt, %s" % col))
        b.append(Pl.curve(lx, np.log10(pr), "line width=0.8pt, %s, dash pattern=on 3pt off 2pt" % col))
        b += ["\\fill[%s] %s circle (1.7pt);" % (col, Pl.pt(x, np.log10(y))) for x, y in zip(lx, li)]
        b += ["\\draw[%s, fill=white, line width=0.6pt] %s circle (1.3pt);" % (col, Pl.pt(x, np.log10(y))) for x, y in zip(lx, pr)]
    # slope triangle above the reference line: one decade in N, half a decade in the error
    x0, y0 = 1.8, ref(1.8) + 0.16
    b.append("\\draw[ink, line width=0.5pt] %s -- %s -- %s -- cycle;" % (Pl.pt(x0, y0), Pl.pt(x0 + 1, y0 - 0.5), Pl.pt(x0 + 1, y0)))
    b.append("\\node[font=\\scriptsize, anchor=south] at %s {$1$};" % Pl.pt(x0 + 0.5, y0))
    b.append("\\node[font=\\scriptsize, anchor=west] at %s {$\\tfrac12$};" % Pl.pt(x0 + 1, y0 - 0.25))
    # notes in place of a legend
    b.append("\\node[note, anchor=west, text=ndInk!60] at %s {$1/\\sqrt N$};" % Pl.pt(4.05, ref(4.0) + 0.04))
    b.append("\\node[note, anchor=south, text=ndGold!80!ndInk] (m) at %s {mixed state, $\\abs{\\Bloch}=0.5$};" % Pl.pt(3.45, -1.22))
    b.append("\\draw[pointer] (m.south) to[bend right=10] %s;" % Pl.pt(3.5, np.interp(3.5, lx, np.log10(li_m)) + 0.02))
    b.append("\\node[note, anchor=north, text=ndBlue] (p) at %s {pure state, $\\abs{\\Bloch}=1$};" % Pl.pt(3.35, -2.3))
    b.append("\\draw[pointer] (p.north) to[bend left=10] %s;" % Pl.pt(3.3, np.interp(3.3, lx, np.log10(li_p)) - 0.02))
    b.append("\\node[note, anchor=north west, align=left] (d) at %s {open marks, dashed: projected onto\\\\ the ball; this helps only a pure state,\\\\"
             " and only for few shots ($%d\\%%$ at $N=%d$)};" % (Pl.pt(1.0, -1.72), round(100 * (1 - pr_p[0] / li_p[0])), Ns[0]))
    b.append("\\draw[pointer] (d.north west) to[bend left=20] %s;" % Pl.pt(0.98, np.log10(pr_p[0]) - 0.03))
    N.write("ch01_tomography_shots", b)
    print("written figures/ch01_tomography_shots.tex; gain of the projection, pure state:",
          np.round(1 - pr_p / li_p, 3))


if __name__ == "__main__":
    main()
