#!/usr/bin/env python3
"""Merge refs_resolved.bib (output of resolve_bib.py) into refs.bib, replacing
the converted @misc entries with the same keys.  Run from the course root."""
import re
src = open("refs.bib", encoding="utf-8").read()
res = open("refs_resolved.bib", encoding="utf-8").read()

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
        yield m.group(2), text[j:p+1]
        i = p + 1

new = {k: e for k, e in entries(res)}
out, replaced = [], 0
pos = 0
for k, e in entries(src):
    start = src.find(e, pos)
    out.append(src[pos:start])
    if k in new and "Converted from the thebibliography" in e:
        note = re.search(r"note = \{Converted from the thebibliography of ([^}]*)\}", e).group(1)
        rep = new[k].rstrip()
        # keep a trace of the origin, and drop DataCite's odd "keywords"/"copyright" noise
        rep = re.sub(r",\s*keywords = \{[^}]*\}", "", rep)
        rep = re.sub(r",\s*copyright = \{[^}]*\}", "", rep)
        rep = rep[:-1].rstrip().rstrip(",") + ",\n  note = {Resolved via Crossref/DataCite from the reference list of %s}\n}" % note
        out.append(rep); replaced += 1
    else:
        out.append(e)
    pos = start + len(e)
out.append(src[pos:])
open("refs.bib", "w", encoding="utf-8").write("".join(out))
print("replaced", replaced, "entries")
