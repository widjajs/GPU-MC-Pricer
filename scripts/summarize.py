#!/usr/bin/env python3
"""Turn a results/<label>.csv from bench.sh into markdown tables (stdlib only)."""
import csv
import sys
from collections import OrderedDict


def main(path: str) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))

    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        vr = "+".join(n for n, on in (("antithetic", r["antithetic"]), ("cv", r["cv"])) if on == "1")
        unit = "d" if "basket" in r["product"] else "steps"
        key = f'{r["product"]} {r["cp"]}, {unit}={r["steps_or_dim"]}, {int(r["paths"]):,} paths' + (
            f", {vr}" if vr else "")
        groups.setdefault(key, []).append(r)

    for key, grp in groups.items():
        cpu = next((float(r["seconds"]) for r in grp if r["backend"].startswith("cpu") and r["backend"] != "cpu1"), None)
        ref = grp[0]["reference"]
        print(f"\n**{key}**" + (f" (closed form {float(ref):.6f})" if ref != "NA" else ""))
        print("\n| backend | price | std err | total ms | kernel ms | Mpaths/s | vs multicore CPU |")
        print("|---|---|---|---|---|---|---|")
        for r in grp:
            secs = float(r["seconds"])
            kms = float(r["kernel_seconds"]) * 1e3
            speed = f"{cpu / secs:.1f}x" if cpu else "-"
            print(f'| {r["backend"]} | {float(r["price"]):.6f} | {float(r["std_error"]):.6f} | '
                  f'{secs * 1e3:.2f} | {f"{kms:.2f}" if kms > 0 else "-"} | {float(r["mpaths_per_s"]):.1f} | {speed} |')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: summarize.py results/<label>.csv")
    main(sys.argv[1])
