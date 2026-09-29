"""ch01_tomography_shots.py -- error of qubit state tomography versus the number
of shots: linear inversion, its projection onto the Bloch ball, and the
1/sqrt(N) reference; also the same for the pure and a mixed state.
Writes figures/ch01_tomography_shots.tex."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tikzexport as T

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
Ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
li_p, pr_p = run(np.array([0.0, 0.0, 1.0]), Ns)      # pure state
li_m, pr_m = run(np.array([0.3, 0.0, 0.4]), Ns)      # mixed state
series = [dict(x=Ns, y=li_p, label="pure state, linear inversion", style="mark=*"),
          dict(x=Ns, y=pr_p, label="pure state, projected", style="mark=*, dashed"),
          dict(x=Ns, y=li_m, label="mixed state, linear inversion", style="mark=square*"),
          dict(x=Ns, y=pr_m, label="mixed state, projected", style="mark=square*, dashed"),
          dict(x=Ns, y=1.0 / np.sqrt(Ns), label=r"$1/\sqrt{N}$", style="gray, thin")]
body = T.pgfplots_axis(series, xlabel="shots per axis $N$", ylabel="mean trace distance to $\\rho$",
                       width="0.85\\linewidth", height="0.45\\linewidth", legend_pos="south west",
                       axis_options="xmode=log, ymode=log")
T.write_tikz("ch01_tomography_shots", body)
print("written figures/ch01_tomography_shots.tex")
