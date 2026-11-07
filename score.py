"""Scores parse_rfq() from parse.py against a labelled .jsonl file.

Usage:  python score.py            (uses my_tests.jsonl)
        python score.py other.jsonl
"""
import json, sys, time
from parse import parse_rfq

FIELDS = ["category", "quantity", "unit", "gsm", "micron", "location"]


def same(field, got, want):
    if want is None or got is None:
        return got is None and want is None
    if field == "quantity":
        try:
            return abs(float(got) - float(want)) < 1e-9
        except (TypeError, ValueError):
            return False
    if field == "location":
        return str(got).strip().lower() == str(want).strip().lower()
    return got == want


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    per_field = {f: 0 for f in FIELDS}
    all_ok = 0
    start = time.time()
    for r in rows:
        out = parse_rfq(r["text"])
        if not isinstance(out, dict):
            raise TypeError("parse_rfq must return a dict")
        ok = True
        for f in FIELDS:
            if same(f, out.get(f), r["expected"][f]):
                per_field[f] += 1
            else:
                ok = False
        all_ok += ok
    secs = time.time() - start
    n = len(rows)
    print(f"Examples: {n}   time: {secs:.2f}s")
    for f in FIELDS:
        print(f"  {f:10s} {per_field[f]:3d}/{n}  {per_field[f] / n:6.1%}")
    print(f"  {'ALL FIELDS':10s} {all_ok:3d}/{n}  {all_ok / n:6.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "my_tests.jsonl")
