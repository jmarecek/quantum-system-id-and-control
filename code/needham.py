"""needham.py -- shared visual style for the geometric figures of the course.

The style is modelled on the drawings in T. Needham, "Visual Differential
Geometry and Forms": crisp ink line art, pale colour washes for regions,
bold labelled vectors, and short italic annotations with thin pointer arrows
that say in words what the picture shows.  (Only the style is borrowed; no
figure of the book is reproduced.)

Usage:
    import needham as N
    body = [...TikZ commands using the styles below...]
    N.write("ch02_name", body)

Styles available inside the tikzpicture:
    ink      main line work              faint   construction / axes
    vec      bold vector with arrow      note    italic annotation text
    pointer  thin arrow from a note      dot     filled point
    wash     pale fill for regions       dashedink  hidden / auxiliary line
Colours: ndInk, ndPaper, ndSky, ndRose, ndBlue, ndRed, ndGold, ndGreen.
"""
import numpy as np
import tikzexport as T

COLOURS = [
    ("ndInk", "0.13,0.13,0.18"),
    ("ndPaper", "0.99,0.96,0.88"),
    ("ndSky", "0.84,0.91,0.98"),
    ("ndRose", "0.98,0.86,0.84"),
    ("ndBlue", "0.13,0.36,0.66"),
    ("ndRed", "0.78,0.20,0.16"),
    ("ndGold", "0.88,0.62,0.10"),
    ("ndGreen", "0.18,0.52,0.30"),
]

STYLES = ", ".join([
    "ink/.style={line width=0.9pt, ndInk}",
    "faint/.style={line width=0.4pt, ndInk!35}",
    "dashedink/.style={line width=0.5pt, ndInk!55, dash pattern=on 2.5pt off 2pt}",
    "vec/.style={->, >=stealth, line width=1.2pt}",
    "note/.style={font=\\footnotesize\\itshape, text=ndInk!85, align=center}",
    "pointer/.style={->, >=stealth, line width=0.4pt, ndInk!60, shorten >=2pt, shorten <=1pt}",
    "dot/.style={circle, fill=ndInk, inner sep=0pt, minimum size=3.4pt}",
    "wash/.style={fill=ndSky, fill opacity=0.55}",
    "every node/.style={text=ndInk}",
])


def write(name, body):
    """Write figures/<name>.tex with the Needham colours and styles."""
    header = "\n".join("\\definecolor{%s}{rgb}{%s}" % c for c in COLOURS)
    text = header + "\n" + "\n".join(b for b in body if b)
    return T.write_tikz(name, text, options=STYLES)


def pt(p):
    return "(%.3f,%.3f)" % (p[0], p[1])


def polyline(points, style):
    return "\\draw[%s] %s;" % (style, " -- ".join(pt(p) for p in points))


class View:
    """Orthographic projection of R^3 onto the page.

    The viewing direction has azimuth `az` and elevation `el` (degrees);
    `R` scales, `shift` translates on the page.
    """

    def __init__(self, az=25.0, el=20.0, R=1.0, shift=(0.0, 0.0)):
        a, e = np.radians(az), np.radians(el)
        self.e1 = np.array([-np.sin(a), np.cos(a), 0.0])
        self.e2 = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
        self.d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.R, self.shift = R, np.asarray(shift, dtype=float)

    def P(self, v):
        v = np.asarray(v, dtype=float)
        return self.shift + self.R * np.array([v @ self.e1, v @ self.e2])

    def pt(self, v):
        return pt(self.P(v))

    def visible(self, v):
        return np.asarray(v, dtype=float) @ self.d >= 0

    def curve(self, pts, front_style, back_style=None):
        """Draw a curve on the unit sphere, solid in front and (optionally) dashed behind."""
        pts = np.asarray(pts, dtype=float)
        mask = pts @ self.d >= -1e-9
        out = []
        for m, style in [(mask, front_style), (~mask, back_style)]:
            if style is None:
                continue
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            for seg in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                if len(seg) > 1:
                    out.append(polyline([self.P(p) for p in pts[seg]], style))
        return out

    def sphere(self, shade=True):
        """Outline of the unit sphere (a circle in orthographic projection)."""
        c, r = self.shift, self.R
        out = []
        if shade:
            out.append("\\shade[ball color=ndSky!70!white, opacity=0.45] %s circle (%.3f);"
                       % (pt(c), r))
        out.append("\\draw[ink] %s circle (%.3f);" % (pt(c), r))
        return out

    def great_circle(self, normal, front="faint", back="faint, densely dotted", n=241):
        """The great circle perpendicular to `normal`."""
        nrm = np.asarray(normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)
        a = np.cross(nrm, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(nrm, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(nrm, a)
        t = np.linspace(0, 2 * np.pi, n)
        pts = np.outer(np.cos(t), a) + np.outer(np.sin(t), b)
        return self.curve(pts, front, back)

    def small_circle(self, axis, polar, front, back=None, n=241):
        """The circle at angular distance `polar` from the unit vector `axis`."""
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        a = np.cross(ax, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(ax, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(ax, a)
        t = np.linspace(0, 2 * np.pi, n)
        pts = (np.cos(polar) * ax[None, :]
               + np.sin(polar) * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b)))
        return self.curve(pts, front, back)


class Plot:
    """A hand-drawn style line plot: page = (x0 + sx (t - tmin), y0 + sy (v - vmin))."""

    def __init__(self, xlim, ylim, width, height, origin=(0.0, 0.0)):
        self.xlim, self.ylim = xlim, ylim
        self.sx = width / (xlim[1] - xlim[0])
        self.sy = height / (ylim[1] - ylim[0])
        self.o = np.asarray(origin, dtype=float)

    def P(self, t, v):
        return self.o + np.array([self.sx * (t - self.xlim[0]), self.sy * (v - self.ylim[0])])

    def pt(self, t, v):
        return pt(self.P(t, v))

    def axes(self, xlabel="", ylabel="", xticks=(), yticks=(), yzero=None):
        """Axis arrows (the x-axis at height yzero, default ylim[0]) and ticks given as (value, label)."""
        yz = self.ylim[0] if yzero is None else yzero
        out = ["\\draw[faint, ->, >=stealth] %s -- %s node[right, text=ndInk] {%s};"
               % (self.pt(self.xlim[0], yz), self.pt(self.xlim[1] + 0.04 * (self.xlim[1] - self.xlim[0]), yz), xlabel),
               "\\draw[faint, ->, >=stealth] %s -- %s node[above, text=ndInk] {%s};"
               % (self.pt(self.xlim[0], self.ylim[0]), self.pt(self.xlim[0], self.ylim[1] + 0.06 * (self.ylim[1] - self.ylim[0])), ylabel)]
        for v, lab in xticks:
            p = self.P(v, yz)
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[below, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0], p[1] + 0.06, p[0], p[1] - 0.06, lab))
        for v, lab in yticks:
            p = self.P(self.xlim[0], v)
            out.append("\\draw[faint] (%.3f,%.3f) -- (%.3f,%.3f) node[left, font=\\scriptsize, text=ndInk] {%s};"
                       % (p[0] + 0.06, p[1], p[0] - 0.06, p[1], lab))
        return out

    def curve(self, ts, vs, style):
        return polyline([self.P(t, v) for t, v in zip(ts, vs)], style)


def ellipsoid_outline(V, centre, axes, n=60):
    """Page polygon of the outline of an axis-aligned ellipsoid (convex hull of its projection)."""
    from scipy.spatial import ConvexHull
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    pts = np.stack([np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)], -1).reshape(-1, 3)
    P2 = np.array([V.P(np.asarray(centre) + np.asarray(axes) * p) for p in pts])
    hull = ConvexHull(P2)
    return [P2[i] for i in hull.vertices]
