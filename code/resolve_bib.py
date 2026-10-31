#!/usr/bin/env python3
"""Resolve the @misc entries converted from thebibliography lists into proper
BibTeX via Crossref (and DataCite for arXiv DOIs).  Writes refs_resolved.bib
(entries to replace, same keys) and refs_unresolved.txt.  Run from the course root.
"""
import re, json, time, sys, os
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote

UA = "qoc-course-bib/1.0 (course materials; no mailto)"
src = open("refs.bib", encoding="utf-8").read()

def entries(text):
    i = 0
    while True:
        j = text.find("@", i)
        if j < 0: return
        m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[j:])
        if not m: i = j + 1; continue
        depth = 0; p = text.find("{", j)
        while p < len(text):
            if text[p] == "{": depth += 1
            elif text[p] == "}":
                depth -= 1
                if depth == 0: break
            p += 1
        yield m.group(1), m.group(2), text[j:p+1]
        i = p + 1

def get(url, accept=None, tries=3):
    for k in range(tries):
        try:
            req = Request(url, headers={"User-Agent": UA, **({"Accept": accept} if accept else {})})
            return urlopen(req, timeout=30).read().decode("utf-8", "replace")
        except Exception as e:
            err = e; time.sleep(2 * (k + 1))
    raise err

def strip_latex(t):
    t = re.sub(r"\\(newblock|em|it|bf|url|href|emph|textit|textbf)\b\{?", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)
    t = re.sub(r"[{}~$\\]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def words(t):
    return set(w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 3)

def bibtex_for_doi(doi, key):
    bib = get("https://doi.org/" + quote(doi), accept="application/x-bibtex").strip()
    m = re.match(r"\s*@(\w+)\s*\{\s*([^,]+),", bib)
    if not m: return None
    bib = bib[:m.start(2)] + key + bib[m.end(2):]
    bib = bib.replace("month=June", "month=jun").replace("month=Nov", "month=nov")
    bib = re.sub(r"month=([A-Z][a-z]{2})[a-z]*", lambda mm: "month=" + mm.group(1).lower(), bib)
    return bib

resolved, unresolved = [], []
todo = [(t, k, e) for t, k, e in entries(src) if t.lower() == "misc" and "Converted from the thebibliography" in e]
print("converted entries:", len(todo), flush=True)
for n, (typ, key, entry) in enumerate(todo):
    m = re.search(r"title\s*=\s*\{(.*)\},\s*note", entry, re.S)
    text = m.group(1) if m else entry
    plain = strip_latex(text)
    doi = None; how = ""
    md = re.search(r"10\.\d{4,9}/[^\s}\]\)\",]+", text)
    ma = re.search(r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})", text, re.I)
    if md:
        doi = md.group(0).rstrip("."); how = "doi-in-text"
    elif ma:
        doi = "10.48550/arXiv." + ma.group(1); how = "arxiv-in-text"
    else:
        try:
            q = urlencode({"query.bibliographic": plain[:300], "rows": "1"})
            d = json.loads(get("https://api.crossref.org/works?" + q))
            items = d["message"]["items"]
        except Exception as e:
            items = []; print("  crossref error", key, e, flush=True)
        if items:
            it = items[0]
            title = (it.get("title") or [""])[0]
            tw = words(strip_latex(title)); pw = words(plain)
            overlap = len(tw & pw) / max(1, len(tw))
            year = str(it.get("issued", {}).get("date-parts", [[None]])[0][0])
            years_in_text = re.findall(r"\b(19\d\d|20\d\d)\b", plain)
            year_ok = (not years_in_text) or (year in years_in_text) or (str(int(year) - 1) in years_in_text if year.isdigit() else False)
            fam = [a.get("family", "") for a in it.get("author", []) if a.get("family")]
            auth_ok = (not fam) or any(f.lower().split()[-1] in plain.lower() for f in fam[:3])
            if overlap >= 0.8 and year_ok and auth_ok and len(tw) >= 2:
                doi = it["DOI"]; how = "crossref score=%.2f" % overlap
        time.sleep(0.5)
    if doi:
        try:
            bib = bibtex_for_doi(doi, key)
        except Exception as e:
            bib = None; print("  doi fetch error", key, doi, e, flush=True)
        if bib:
            resolved.append((key, how, bib)); print("%3d OK   %-28s %s  [%s]" % (n, key, doi, how), flush=True)
            continue
    unresolved.append((key, plain[:160])); print("%3d ---  %-28s %s" % (n, key, plain[:90]), flush=True)

with open("refs_resolved.bib", "w", encoding="utf-8") as f:
    for key, how, bib in resolved:
        f.write("%% %s resolved via %s\n%s\n\n" % (key, how, bib))
with open("refs_unresolved.txt", "w", encoding="utf-8") as f:
    for key, plain in unresolved:
        f.write("%s\t%s\n" % (key, plain))
print("resolved %d, unresolved %d" % (len(resolved), len(unresolved)))
