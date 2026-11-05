"""Local stand-in for the official score.py (which was not included with the brief).
Usage: python eval_local.py my_tests.jsonl [-v]
Reports exact-match on all six fields and per-field accuracy."""
import json, sys, time
from parse import parse_rfq

FIELDS = ["category", "quantity", "unit", "gsm", "micron", "location"]

def same(k, a, b):
    if k == "location":
        return (a or "").strip().lower() == (b or "").strip().lower()
    if k == "quantity" and a is not None and b is not None:
        return abs(float(a) - float(b)) < 1e-9
    return a == b

def main(path, verbose):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    t0 = time.time(); exact = 0; per = {k: 0 for k in FIELDS}; bad = []
    for r in rows:
        got = parse_rfq(r["text"]); exp = r["expected"]
        assert set(got) == set(FIELDS), got
        oks = {k: same(k, got[k], exp[k]) for k in FIELDS}
        for k in FIELDS: per[k] += oks[k]
        if all(oks.values()): exact += 1
        else: bad.append((r["text"], {k: (got[k], exp[k]) for k in FIELDS if not oks[k]}))
    n = len(rows)
    print(f"{path}: exact-match {exact}/{n} = {exact/n:.1%}   ({time.time()-t0:.2f}s)")
    print("per-field:", {k: f"{v}/{n}" for k, v in per.items()})
    if verbose or bad:
        for t, d in bad: print("  X", t, "\n     ", d)

if __name__ == "__main__":
    main(sys.argv[1], "-v" in sys.argv)
