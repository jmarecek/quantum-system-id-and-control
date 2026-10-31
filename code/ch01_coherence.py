"""ch01_coherence.py -- a quarter of a century of coherence in superconducting qubits.

The lifetimes T1 (energy relaxation) and T2 (coherence) reported for
superconducting qubits, from the first coherent oscillations of a Cooper-pair
box in 1999 to 2025, in microseconds on a logarithmic scale, against the year
of the journal volume in which each result appeared.  The figure brings up to
date Fig. 2c of Kjaergaard et al., Annu. Rev. Condens. Matter Phys. 11, 369
(2020), doi:10.1146/annurev-conmatphys-031119-050605, reproduced as Fig. 1a of
Siddiqi, Nat. Rev. Mater. 6, 875 (2021), doi:10.1038/s41578-021-00370-4, and
keeps its three kinds of qubit:
  * qubits encoded in a circuit of Josephson junctions (dots; open rings for
    gatemons, whose junction is a semiconductor or graphene weak link);
  * qubits encoded in the photons of a superconducting cavity and controlled
    through a junction (squares; T1 and T2 of the single-photon Fock qubit
    |0>, |1> of the storage cavity);
  * error-corrected logical qubits (crosses; the lifetime of the logical
    information with the correction running).
Every value below is taken from the text of the primary paper (the largest the
paper quotes; T2 is whichever of the echo, Ramsey or CPMG times the paper
quotes); where a paper states only one of T1 and T2, only that one is drawn.
Where the values differ from those plotted by Kjaergaard et al. (Ofek et al.,
for instance, give 320 us with error correction, not the 390 us of the review),
the primary paper is followed.  The thin step line is the record so far of
max(T1, T2) among the junction qubits.
Writes figures/ch01_coherence.tex.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import needham as N

# (year, kind, T1 [us], T2 [us], source and the values it quotes); kind is
# "jj", "gatemon", "cavity" or "qec"; None where the paper gives no value.
DATA = [
    (1999, "jj", None, 0.002, "Nakamura, Pashkin & Tsai, Nature 398, 786 (1999), doi:10.1038/19718: oscillations observed up to about 2 ns"),
    (2002, "jj", 1.8, 0.50, "Vion et al., Science 296, 886 (2002), doi:10.1126/science.1069372: T1 = 1.8 us, decoherence time 0.50 us"),
    (2003, "jj", 0.90, 0.020, "Chiorescu et al., Science 299, 1869 (2003), doi:10.1126/science.1081045: T1 = 900 ns, Ramsey 20 ns"),
    (2005, "jj", None, 3.9, "Bertet et al., Phys. Rev. Lett. 95, 257002 (2005), doi:10.1103/PhysRevLett.95.257002: echo 3.9 us"),
    (2008, "jj", 4.0, None, "Houck et al., Phys. Rev. Lett. 101, 080502 (2008), doi:10.1103/PhysRevLett.101.080502: T1 up to 4 us"),
    (2009, "jj", None, 0.35, "Manucharyan et al., Science 326, 113 (2009), doi:10.1126/science.1175552: decoherence time at least 350 ns"),
    (2011, "jj", 12.0, 23.0, "Bylander et al., Nat. Phys. 7, 565 (2011), doi:10.1038/nphys1994: T1 = 12 us, echo 23 us"),
    (2011, "jj", 60.0, 20.0, "Paik et al., Phys. Rev. Lett. 107, 240501 (2011), doi:10.1103/PhysRevLett.107.240501: T1 up to 60 us, T2 10-20 us"),
    (2012, "jj", 70.0, 95.0, "Rigetti et al., Phys. Rev. B 86, 100506(R) (2012), doi:10.1103/PhysRevB.86.100506: T1 = 70 us, Ramsey 95 us"),
    (2013, "jj", 60.0, 60.0, "Chang et al., Appl. Phys. Lett. 103, 012602 (2013), doi:10.1063/1.4813269: T1 ~ T2 ~ 60 us"),
    (2014, "jj", 1000.0, None, "Pop et al., Nature 508, 369 (2014), doi:10.1038/nature13017: T1 in excess of 1 ms "
                               "(as quoted by Vool et al., Phys. Rev. Lett. 113, 247001 (2014), doi:10.1103/PhysRevLett.113.247001)"),
    (2015, "jj", 80.0, 154.0, "Jin et al., Phys. Rev. Lett. 114, 240501 (2015), doi:10.1103/PhysRevLett.114.240501: T1 = 80 us, echo 154 us"),
    (2016, "jj", 55.0, 85.0, "Yan et al., Nat. Commun. 7, 12964 (2016), doi:10.1038/ncomms12964: T1 = 55 us, CPMG 85 us"),
    (2019, "jj", None, 400.0, "Nguyen et al., Phys. Rev. X 9, 041041 (2019), doi:10.1103/PhysRevX.9.041041: T2 above 400 us in one device"),
    (2021, "jj", 360.0, 249.0, "Place et al., Nat. Commun. 12, 1779 (2021), doi:10.1038/s41467-021-22030-5: T1 up to 0.36 ms, echo 249 us"),
    (2022, "jj", 503.0, None, "Wang et al., npj Quantum Inf. 8, 3 (2022), doi:10.1038/s41534-021-00510-2: T1 up to 503 us"),
    (2023, "jj", 1200.0, 1480.0, "Somoroff et al., Phys. Rev. Lett. 130, 267001 (2023), doi:10.1103/PhysRevLett.130.267001: "
                                 "T1 = 1.20 ms (averaged), Ramsey 1.48 ms"),
    (2025, "jj", 1168.0, 943.0, "Wang et al., Phys. Rev. Applied 23, 044064 (2025), doi:10.1103/PhysRevApplied.23.044064: T1 = 1.168 ms, echo 0.943 ms"),
    (2025, "jj", 666.0, 1057.0, "Tuokkola et al., Nat. Commun. 16, 5421 (2025), doi:10.1038/s41467-025-61126-0: T1 up to 666 us, echo up to 1057 us"),
    (2025, "jj", 1682.0, 1185.0, "Bland et al., Nature 647, 343 (2025), doi:10.1038/s41586-025-09687-4: T1 up to 1.68 ms, echo up to 1.18 ms"),
    (2015, "gatemon", 0.83, 0.91, "Larsen et al., Phys. Rev. Lett. 115, 127001 (2015), doi:10.1103/PhysRevLett.115.127001: T1 0.56-0.83 us, Ramsey 0.73-0.91 us"),
    (2016, "gatemon", 5.3, 9.5, "Casparis et al., Phys. Rev. Lett. 116, 150505 (2016), doi:10.1103/PhysRevLett.116.150505: T1 = 5.3 us, echo 9.5 us"),
    (2018, "gatemon", 20.0, 20.0, "Luthi et al., Phys. Rev. Lett. 120, 100502 (2018), doi:10.1103/PhysRevLett.120.100502: T1 and echo 15-20 us"),
    (2019, "gatemon", 0.036, None, "Wang et al., Nat. Nanotechnol. 14, 120 (2019), doi:10.1038/s41565-018-0329-2: graphene junction, T1 about 36 ns"),
    (2008, "cavity", 3.47, 6.53, "Wang et al., Phys. Rev. Lett. 101, 240401 (2008), doi:10.1103/PhysRevLett.101.240401: resonator T1 = 3.47 us, T2 = 6.53 us"),
    (2016, "cavity", 3300.0, 1100.0, "Wang et al., Science 352, 1087 (2016), doi:10.1126/science.aaf2941: cavity T1 2.2-3.3 ms, Ramsey 0.8-1.1 ms"),
    (2016, "cavity", 250.0, 330.0, "Ofek et al., Nature 536, 441 (2016), doi:10.1038/nature18949: cavity T1 ~ 250 us, T2 ~ 330 us"),
    (2018, "cavity", 1100.0, 1900.0, "Rosenblum et al., Science 361, 266 (2018), doi:10.1126/science.aat3996: cavity T1 = 1.1 ms, T2 = 1.9 ms"),
    (2019, "cavity", 143.0, 252.0, "Hu et al., Nat. Phys. 15, 503 (2019), doi:10.1038/s41567-018-0414-3: cavity T1 = 143 us, T2 = 252 us"),
    (2021, "cavity", 2000.0, None, "Chakram et al., Phys. Rev. Lett. 127, 107701 (2021), doi:10.1103/PhysRevLett.127.107701: single-photon lifetimes 2 ms"),
    (2023, "cavity", 25600.0, 34000.0, "Milul et al., PRX Quantum 4, 030336 (2023), doi:10.1103/PRXQuantum.4.030336: T1 = 25.6 ms, T2 = 34 ms"),
    (2023, "cavity", 578.0, 915.0, "Ni et al., Nature 616, 56 (2023), doi:10.1038/s41586-023-05784-4: cavity T1 = 578 us, Ramsey 915 us"),
    (2024, "cavity", 1400.0, 2700.0, "Ganjam et al., Nat. Commun. 15, 3687 (2024), doi:10.1038/s41467-024-47857-6: on-chip memory T1 1.0-1.4 ms, Ramsey 2.0-2.7 ms"),
    (2016, "qec", 320.0, None, "Ofek et al. (above): 320 us with error correction (cat code)"),
    (2019, "qec", 200.0, None, "Hu et al. (above): 200 us with error correction (binomial code)"),
    (2023, "qec", 805.0, None, "Ni et al. (above): 805 us with error correction (binomial code), beyond break-even"),
    (2023, "qec", 1820.0, None, "Sivak et al., Nature 616, 50 (2023), doi:10.1038/s41586-023-05782-6: 1.82 ms with error correction (GKP code), beyond break-even"),
    (2025, "qec", 291.0, None, "Google Quantum AI, Nature 638, 920 (2025), doi:10.1038/s41586-024-08449-y: 291 us, distance-7 surface code, beyond break-even"),
]


def positions():
    """Horizontal positions: the middle of the year, entries of one year spread a little apart."""
    xs, count = [], {}
    for y, *_ in DATA:
        count[y] = count.get(y, 0) + 1
    seen = {}
    for y, *_ in DATA:
        k, n = seen.get(y, 0), count[y]
        seen[y] = k + 1
        xs.append(y + 0.5 + (0.0 if n == 1 else 0.62 * (k / (n - 1) - 0.5)))
    return np.array(xs)


def record():
    """The record so far of max(T1, T2) among the junction qubits, as (year, value) steps."""
    best, steps = 0.0, []
    for y, kind, t1, t2, _ in sorted(DATA, key=lambda d: d[0]):
        v = max(t for t in (t1, t2) if t is not None)
        if kind == "jj" and v > best:
            best = v
            steps.append((y, v))
    return steps


def main():
    xs = positions()
    steps = record()
    print("record so far (year, us):", steps)
    # the pace: tenfold every few years up to the millisecond, very little since
    (ya, va), (yb, vb) = steps[0], [(y, v) for y, v in steps if y <= 2014][-1]
    per_decade = (yb - ya) / np.log10(vb / va)
    gain_post = steps[-1][1] / vb
    print("%d-%d: tenfold every %.1f years; since then a further factor %.1f" % (ya, yb, per_decade, gain_post))
    assert 2.0 < per_decade < 3.5 and gain_post < 2.0
    jj = [d for d in DATA if d[1] == "jj"]
    assert max(max(t for t in d[2:4] if t) for d in jj) / min(max(t for t in d[2:4] if t) for d in jj) > 1e5   # five orders

    b = []
    Pl = N.Plot((1998.0, 2026.6), (-3.0, 5.0), 12.6, 5.8, origin=(0.0, 0.0))
    lab = {-3: "1\\,ns", -2: "10\\,ns", -1: "100\\,ns", 0: "1\\,$\\mu$s", 1: "10\\,$\\mu$s", 2: "100\\,$\\mu$s", 3: "1\\,ms", 4: "10\\,ms", 5: "100\\,ms"}
    b += Pl.axes("year", "lifetime", xticks=[(y, "$%d$" % y) for y in (2000, 2005, 2010, 2015, 2020, 2025)],
                 yticks=[(k, lab[k]) for k in range(-3, 6)])
    for k in range(-2, 6):
        b.append("\\draw[faint, ndInk!12] %s -- %s;" % (Pl.pt(1998.0, k), Pl.pt(2026.6, k)))
    # the record so far, a thin step line
    path = []
    for (y0, v0), (y1, _) in zip(steps, steps[1:] + [(2026.4, None)]):
        path += [Pl.P(y0 + 0.5, np.log10(v0)), Pl.P(y1 + 0.5 if y1 < 2026 else y1, np.log10(v0))]
    b.append(N.polyline(path, "line width=0.6pt, ndInk!55"))
    # markers: T1 red, T2 blue; dots, rings, squares and crosses by kind
    for x, (y, kind, t1, t2, _) in zip(xs, DATA):
        if t1 is not None and t2 is not None:
            b.append("\\draw[faint] %s -- %s;" % (Pl.pt(x, np.log10(t1)), Pl.pt(x, np.log10(t2))))
        for t, col in ((t1, "ndRed"), (t2, "ndBlue")):
            if t is None:
                continue
            p = Pl.pt(x, np.log10(t))
            if kind == "jj":
                b.append("\\fill[%s] %s circle (2.0pt);" % (col, p))
            elif kind == "gatemon":
                b.append("\\draw[%s, line width=0.7pt, fill=white] %s circle (1.7pt);" % (col, p))
            elif kind == "cavity":
                b.append("\\fill[%s] %s ++(-2.0pt,-2.0pt) rectangle ++(4.0pt,4.0pt);" % (col, p))
            else:
                b.append("\\draw[ndGreen, line width=1.3pt] %s ++(-2.6pt,-2.6pt) -- ++(5.2pt,5.2pt) ++(-5.2pt,0) -- ++(5.2pt,-5.2pt);" % p)
    # legend, under the axis
    leg = [("\\fill[ndRed] (0,0) circle (2.0pt);", "$T_1$, energy relaxation"),
           ("\\fill[ndInk] (0,0) circle (2.0pt);", "qubit in a circuit of junctions"),
           ("\\fill[ndInk] (-2.0pt,-2.0pt) rectangle (2.0pt,2.0pt);", "qubit in the photons of a cavity"),
           ("\\fill[ndBlue] (0,0) circle (2.0pt);", "$T_2$, coherence"),
           ("\\draw[ndInk, line width=0.7pt] (0,0) circle (1.7pt);", "the same, with a gatemon junction"),
           ("\\draw[ndGreen, line width=1.3pt] (-2.6pt,-2.6pt) -- (2.6pt,2.6pt) (-2.6pt,2.6pt) -- (2.6pt,-2.6pt);", "logical qubit, error corrected")]
    for k, (mark, text) in enumerate(leg):
        x, y = (0.25, 3.7, 9.4)[k % 3], -1.0 - 0.48 * (k // 3)
        b.append("\\begin{scope}[shift={(%.3f,%.3f)}]%s\\end{scope}" % (x, y, mark))
        b.append("\\node[anchor=west, font=\\footnotesize] at (%.3f,%.3f) {%s};" % (x + 0.12, y, text))
    # notes on the families
    P = lambda yr, us: Pl.pt(yr, np.log10(us))
    b.append("\\node[note, anchor=south west] at %s {Cooper-pair box};" % P(1999.8, 0.0023))
    b.append("\\node[note, anchor=east] at %s {quantronium};" % P(2002.25, 1.8))
    b.append("\\node[note, anchor=west] at %s {flux qubits};" % P(2003.85, 0.020))
    b.append("\\node[note, anchor=north west] (fl) at %s {fluxonium};" % P(2009.0, 0.14))
    b.append("\\draw[pointer] (fl.north west) to[bend left=25] %s;" % P(2009.42, 0.30))
    b.append("\\node[note, anchor=north, text=ndInk!70] (gm) at %s {gatemons};" % P(2017.4, 0.06))
    b.append("\\draw[pointer] (gm.north) to[bend right=20] %s;" % P(2015.35, 0.70))
    b.append("\\draw[pointer] (gm.east) to[bend right=25] %s;" % P(2019.35, 0.040))
    b.append("\\node[note, anchor=north] (tr) at %s {transmons\\\\ in 3D cavities};" % P(2013.0, 6.5))
    b.append("\\draw[pointer] (tr.north) to[bend right=10] %s;" % P(2012.5, 50.0))
    b.append("\\node[note, anchor=north] (ta) at %s {tantalum\\\\ transmons};" % P(2021.9, 150.0))
    b.append("\\draw[pointer] (ta.north) to[bend right=10] %s;" % P(2021.75, 300.0))
    b.append("\\node[note, anchor=south, text=ndInk] (cv) at %s {cavity qubits};" % P(2020.0, 30000.0))
    b.append("\\draw[pointer] (cv.east) to[bend left=15] %s;" % P(2023.1, 28000.0))
    b.append("\\node[note, anchor=north, text=ndGreen] (qe) at %s {error correction\\\\ beyond break-even};" % P(2023.9, 14.0))
    b.append("\\draw[pointer] (qe.north) to[bend right=15] %s;" % P(2023.75, 650.0))
    b.append("\\draw[pointer] (qe.north east) to[bend right=20] %s;" % P(2025.7, 240.0))
    b.append("\\node[note, anchor=south] (rc) at %s {the record so far};" % P(2006.4, 14.0))
    b.append("\\draw[pointer] (rc.south) to[bend right=15] %s;" % P(2007.0, 4.3))
    b.append("\\node[note, anchor=east] at %s {fluxonium, 1\\,ms};" % P(2014.1, 1000.0))
    b.append("\\node[note, anchor=north west, align=left] at %s {from %d to %d tenfold every $%.1f$ years;\\\\ since then less than twofold};"
             % (P(1998.6, 6.0e4), ya, yb, per_decade))
    N.write("ch01_coherence", b)
    print("written figures/ch01_coherence.tex; %d entries" % len(DATA))


if __name__ == "__main__":
    main()
