"""ch01_funding.py -- where the money goes: funding of the qubit platforms.

A header row of small cartoons of the five platforms (a photonic waveguide
chip, a superconducting circuit with its junction, a spin in a quantum dot,
atoms in optical tweezers, a chain of trapped ions), each with a line saying
what the qubit is, and below it
  (left)  the funding of each platform over two windows, as published by
          McKinsey in two successive editions of its Quantum Technology
          Monitor: 2023-24 (pale) and 2023-25 (solid), in $ billion;
  (right) the investment in all quantum-technology start-ups, year by year,
          2022-25, in $ billion.
The windows overlap and are not added; the difference between them is
nearly all 2025, as the note on the trapped ions checks against the four
largest trapped-ion deals of that year.

Data (only numbers read in the sources):
  [M25] McKinsey & Company, "The year of quantum: from concept to reality in
        2025", June 2025, exhibit "Funding, 2023-24, $ million" (the image
        figures/hw_mckinsey_2025.png): photonic networks 886, superconducting
        circuits 1,033, spin qubits 312, neutral atoms 386 (including $230M
        to QuEra at the start of 2025), trapped ions 339.
        https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/the-year-of-quantum-from-concept-to-reality-in-2025
  [M26] McKinsey & Company, "Quantum Technology Monitor 2026", April 2026,
        p. 50, "Funding, 2023-25, $ billion": photonic 2.8, superconducting
        2.4, spin 0.4, neutral atoms 0.7, trapped ions 4.6; footnote: assumes
        $100M a year for each major superconducting player (Alibaba, AWS,
        Google, IBM) and $50M a year for spin qubits (Intel); sources Capital
        IQ, Crunchbase, PitchBook, Quantum Computing Report.
        p. 36 (PitchBook), all QT start-ups, $ billion: 2022 2.3, 2023 1.3,
        2024 2.0, 2025 12.6.
        p. 37, largest 2025 deals, $ million: IonQ 1,535 and 786 (public
        offerings), Oxford Ionics 1,075 (reverse merger with IonQ),
        Quantinuum 839, PsiQuantum 750.
        https://www.mckinsey.com/capabilities/mckinsey-technology/our-insights/mckinsey-quantum-technology-monitor-2026-a-commercial-tipping-point
Writes figures/ch01_funding.tex.
"""
import numpy as np
import needham as N

PLATFORMS = ["Photonic", "Superconducting", "Spin qubits", "Neutral atoms", "Trapped ions"]
WHAT = ["a photon in one\\\\ of two waveguides",
        "two levels of a\\\\ junction circuit",
        "an electron's spin\\\\ in a quantum dot",
        "two levels of an\\\\ atom in a tweezer",
        "two levels of an\\\\ ion in a trap"]
F2324 = np.array([886, 1033, 312, 386, 339]) / 1000.0        # [M25], $ billion
F2325 = np.array([2.8, 2.4, 0.4, 0.7, 4.6])                  # [M26] p. 50
YEARS = [2022, 2023, 2024, 2025]
TOTAL = np.array([2.3, 1.3, 2.0, 12.6])                      # [M26] p. 36
IONS_2025 = {"IonQ": 1.535 + 0.786, "Oxford Ionics": 1.075, "Quantinuum": 0.839}   # [M26] p. 37

COL = 2.45                                                  # width of a platform column, cm
SB = 0.72                                                    # cm per $ billion, platforms
ST = 3.3 / 12.6                                              # cm per $ billion, all start-ups
OLD = "fill=ndSky, draw=ndBlue!70, line width=0.5pt"
NEW = "fill=ndBlue!80, draw=ndInk, line width=0.5pt"


def chip(b, x, y):
    """Two waveguides meeting in a coupler on a chip, a photon in the upper one."""
    b.append("\\filldraw[fill=ndSky!60, draw=ndInk, line width=0.6pt, rounded corners=1.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x - 0.55, y - 0.36, x + 0.55, y + 0.36))
    s = np.linspace(-0.55, 0.55, 80)
    for sg in (1, -1):
        b.append(N.polyline([(x + t, y + sg * (0.17 - 0.12 * np.exp(-(t / 0.13) ** 2))) for t in s], "ndInk, line width=0.7pt"))
    s = np.linspace(-0.47, -0.17, 60)
    b.append(N.polyline([(x + t, y + 0.17 + 0.07 * np.sin(2 * np.pi * (t + 0.32) / 0.07) * np.exp(-((t + 0.32) / 0.07) ** 2)) for t in s], "ndRed, line width=0.7pt"))


def junction(b, x, y):
    """Two capacitor pads joined through a Josephson junction (the cross)."""
    for sg in (-1, 1):
        b.append("\\filldraw[fill=ndGold!55, draw=ndInk, line width=0.6pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x + sg * 0.17, y - 0.3, x + sg * 0.52, y + 0.3))
    b.append("\\draw[ndInk, line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (x - 0.17, y, x + 0.17, y))
    b.append("\\draw[ndInk, line width=0.9pt] (%.3f,%.3f) -- (%.3f,%.3f) (%.3f,%.3f) -- (%.3f,%.3f);" % (x - 0.07, y - 0.07, x + 0.07, y + 0.07, x - 0.07, y + 0.07, x + 0.07, y - 0.07))


def dot_spin(b, x, y):
    """Gate electrodes above a potential well that holds one electron with its spin."""
    for dx in (-0.36, 0.0, 0.36):
        b.append("\\filldraw[fill=ndInk!25, draw=ndInk, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x + dx - 0.12, y + 0.22, x + dx + 0.12, y + 0.38))
    s = np.linspace(-0.55, 0.55, 80)
    b.append(N.polyline([(x + t, y - 0.02 - 0.3 * np.exp(-(t / 0.2) ** 2)) for t in s], "ndInk, line width=0.7pt"))
    b.append("\\fill[ndBlue] (%.3f,%.3f) circle (2.2pt);" % (x, y - 0.25))
    b.append("\\draw[->, >=stealth, ndRed, line width=1.0pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (x, y - 0.33, x, y + 0.1))


def tweezers(b, x, y):
    """Atoms held at the waists of focused laser beams."""
    h = np.linspace(-0.38, 0.38, 30)
    for dx in (-0.36, 0.0, 0.36):
        w = 0.035 + 0.13 * np.abs(h) / 0.38
        poly = [(x + dx - wi, y + hi) for wi, hi in zip(w, h)] + [(x + dx + wi, y + hi) for wi, hi in zip(w[::-1], h[::-1])]
        b.append("\\fill[ndRose] %s -- cycle;" % " -- ".join(N.pt(p) for p in poly))
        b.append("\\fill[ndBlue] (%.3f,%.3f) circle (2.2pt);" % (x + dx, y))


def ions(b, x, y):
    """A chain of ions between the rods and end caps of a linear trap, one of them lit by a laser."""
    for sg in (-1, 1):
        b.append("\\draw[faint, line width=1.4pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (x - 0.5, y + sg * 0.27, x + 0.5, y + sg * 0.27))
        b.append("\\filldraw[fill=ndInk!25, draw=ndInk, line width=0.5pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x + sg * 0.5, y - 0.15, x + sg * 0.6, y + 0.15))
    pos = np.array([-1.437, -0.757, 0.0, 0.757, 1.437]) * 0.28          # equilibrium of five ions in a harmonic trap
    b.append("\\draw[ndRed!70, line width=1.2pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (x + pos[3] - 0.16, y - 0.42, x + pos[3], y))
    for p in pos:
        b.append("\\fill[ndBlue] (%.3f,%.3f) circle (2.2pt);" % (x + p, y))


def main():
    for k in range(4):
        assert abs(TOTAL[k] - (2.3, 1.3, 2.0, 12.6)[k]) < 1e-12
    assert abs(TOTAL[3] / TOTAL[2] - 6.3) < 0.01                                          # [M26]: "6.3x year over year"
    assert abs((F2325[4] - F2324[4]) - sum(IONS_2025.values())) < 0.1                     # the trapped-ion difference is 2025
    b = []
    # ---- header: cartoons, names, what the qubit is
    for k, (name, what, icon) in enumerate(zip(PLATFORMS, WHAT, (chip, junction, dot_spin, tweezers, ions))):
        cx = COL * (k + 0.5)
        icon(b, cx, 6.2)
        b.append("\\node[font=\\footnotesize\\bfseries] at (%.3f,%.3f) {%s};" % (cx, 5.5, name))
        b.append("\\node[font=\\scriptsize, align=center] at (%.3f,%.3f) {%s};" % (cx, 4.82, what))
    b.append("\\draw[faint] (0.1,4.3) -- (%.3f,4.3);" % (5 * COL - 0.1))
    # ---- funding per platform over the two windows
    b.append("\\draw[faint] (0.1,0) -- (%.3f,0);" % (5 * COL - 0.1))
    for k in range(5):
        cx = COL * (k + 0.5)
        for dx, v, st, fmt in ((-0.3, F2324[k], OLD, "%.2f"), (0.3, F2325[k], NEW, "%.1f")):
            b.append("\\filldraw[%s] (%.3f,0) rectangle (%.3f,%.3f);" % (st, cx + dx - 0.26, cx + dx + 0.26, SB * v))
            b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {%s};" % (cx + dx, SB * v, fmt % v))
    # legend, in the empty space above the spin qubits
    lx, ly = 2 * COL + 0.1, 3.45
    b.append("\\filldraw[%s] (%.3f,%.3f) rectangle ++(0.3,0.2) node[right, font=\\scriptsize, text=ndInk, yshift=-0.1cm] {2023--24 (McKinsey, 2025)};" % (OLD, lx, ly))
    b.append("\\filldraw[%s] (%.3f,%.3f) rectangle ++(0.3,0.2) node[right, font=\\scriptsize, text=ndInk, yshift=-0.1cm] {2023--25 (McKinsey, 2026)};" % (NEW, lx, ly - 0.38))
    b.append("\\node[font=\\scriptsize, anchor=west] at (%.3f,%.3f) {funding of each platform, \\$\\,billion};" % (lx - 0.04, ly + 0.45))
    # what the trapped-ion bar is made of
    top = SB * F2325[4]
    b.append("\\node[note, anchor=east, align=right] (ion) at (%.3f,%.3f) {in 2025 alone: IonQ's two offerings ($2.3$),\\\\ its purchase of Oxford Ionics ($1.1$)\\\\ and Quantinuum's round ($0.8$)};" % (4 * COL + 0.05, 2.35))
    b.append("\\draw[pointer] (ion.east) to[bend left=12] (%.3f,%.3f);" % (4.5 * COL + 0.03, top - 0.6))
    b.append("\\node[note, anchor=south] (ps) at (%.3f,%.3f) {PsiQuantum's\\\\ round, $0.75$\\\\ in 2025};" % (0.5 * COL - 0.45, 2.3))
    b.append("\\draw[pointer] (ps.south) to[bend right=20] (%.3f,%.3f);" % (0.5 * COL + 0.04, SB * F2325[0] - 0.25))
    # ---- all start-ups, year by year
    ox = 5 * COL + 0.55
    b.append("\\draw[faint] (%.3f,4.3) -- (%.3f,0);" % (ox - 0.3, ox - 0.3))
    b.append("\\node[font=\\footnotesize\\bfseries, anchor=west] at (%.3f,%.3f) {All start-ups};" % (ox - 0.05, 5.5))
    b.append("\\node[font=\\scriptsize, align=left, anchor=west] at (%.3f,%.3f) {investment in quantum-\\\\technology start-ups,\\\\ by year, \\$\\,billion};" % (ox - 0.05, 4.82))
    b.append("\\draw[faint] (%.3f,0) -- (%.3f,0);" % (ox, ox + 2.95))
    for k, (yr, v) in enumerate(zip(YEARS, TOTAL)):
        x = ox + 0.35 + 0.74 * k
        b.append("\\filldraw[fill=ndGold!80, draw=ndInk, line width=0.5pt] (%.3f,0) rectangle (%.3f,%.3f);" % (x - 0.22, x + 0.22, ST * v))
        b.append("\\node[font=\\scriptsize, anchor=south] at (%.3f,%.3f) {%s};" % (x, ST * v, "%.1f" % v))
        b.append("\\node[font=\\scriptsize, anchor=north] at (%.3f,-0.04) {%d};" % (x, yr))
    b.append("\\node[note, anchor=south west, align=left] (x6) at (%.3f,%.3f) {$\\times6.3$ in\\\\ one year;\\\\ nine tenths\\\\ for computing};" % (ox - 0.1, 1.05))
    b.append("\\draw[pointer] (x6.east) to[bend left=15] (%.3f,%.3f);" % (ox + 0.35 + 0.74 * 3 - 0.24, ST * TOTAL[3] - 0.7))
    N.write("ch01_funding", b)
    print("written figures/ch01_funding.tex; trapped ions 2023-25 minus 2023-24: %.2f, four 2025 deals: %.2f"
          % (F2325[4] - F2324[4], sum(IONS_2025.values())))


if __name__ == "__main__":
    main()
