#!/usr/bin/env python3
"""Phase 0: Staleness detection for analysis outputs.

Checks whether analysis scripts have been modified more recently than their
output files. If a script changed but output didn't, the output is stale.

Configuration:
    Fill in DEPENDENCIES with your script→output mappings.
    Optionally fill in TEX_NUMBERS with key values to verify in .tex files.

Usage:
    python check_staleness.py                # check freshness
    python check_staleness.py --update-hashes  # record current state as baseline

Exit codes: 0 = all fresh, 1 = stale outputs found
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# ── CONFIGURE THESE ──

PAPER_DIR = Path(__file__).resolve().parent.parent  # adjust to your layout
HASH_CACHE = PAPER_DIR / "provenance" / "file_hashes.json"

# Map each script to its expected outputs.
# Example:
#   {"script": Path("analysis/compute_results.py"),
#    "outputs": [Path("analysis/results/table1.csv")],
#    "label": "Table 1 metrics"}
DEPENDENCIES = [
    # {
    #     "script": PAPER_DIR.parent / "analysis" / "compute_results.py",
    #     "outputs": [
    #         PAPER_DIR.parent / "analysis" / "results" / "table1.csv",
    #     ],
    #     "label": "Main results",
    # },
]

# Key numbers that must appear in .tex files.
# Example:
#   {"file": "main.tex", "value": 87.8, "label": "Directional accuracy"}
TEX_NUMBERS = [
    # {"file": "main.tex", "value": 87.8, "label": "Dir acc"},
]


# ── Helpers ──

def file_hash(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_mtime_str(path):
    if not path.exists():
        return "MISSING"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def load_cache():
    if HASH_CACHE.exists():
        with open(HASH_CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_CACHE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-hashes", action="store_true")
    args = parser.parse_args()

    cache = load_cache()
    stale, missing, fresh = [], [], []

    print("=== Staleness Detection ===\n")

    # Check script→output freshness
    for dep in DEPENDENCIES:
        script = dep["script"]
        if not script.exists():
            print(f"  SKIP {dep['label']}: script missing")
            continue

        script_mtime = script.stat().st_mtime
        script_hash_now = file_hash(script)
        script_hash_cached = cache.get(str(script), {}).get("hash")
        script_changed = script_hash_cached and script_hash_cached != script_hash_now

        for output in dep["outputs"]:
            if not output.exists():
                missing.append(dep["label"])
                print(f"  MISSING {dep['label']}: {output.name}")
                continue

            if script_mtime > output.stat().st_mtime:
                stale.append(f"{dep['label']}: script newer than output")
                print(f"  STALE {dep['label']}: {script.name} > {output.name}")
            elif script_changed:
                stale.append(f"{dep['label']}: script hash changed")
                print(f"  STALE {dep['label']}: hash changed")
            else:
                fresh.append(dep["label"])

    # Check key .tex numbers
    if TEX_NUMBERS:
        print("\n--- Key Numbers ---")
        for check in TEX_NUMBERS:
            tex_path = PAPER_DIR / check["file"]
            with open(tex_path) as f:
                content = f.read()
            found = any(s in content for s in [f"{check['value']:.1f}", f"{check['value']:.0f}"])
            status = "OK" if found else "MISSING"
            print(f"  {status} {check['label']}: {check['value']} in {check['file']}")
            if not found:
                stale.append(f"{check['label']}: value not in tex")

    # Update hashes
    if args.update_hashes:
        new_cache = {}
        for dep in DEPENDENCIES:
            for f in [dep["script"]] + dep["outputs"]:
                if f.exists():
                    new_cache[str(f)] = {"hash": file_hash(f), "checked": datetime.now().isoformat()}
        save_cache(new_cache)
        print(f"\n  Saved {len(new_cache)} hashes to {HASH_CACHE}")

    # Summary
    print(f"\n=== Summary: {len(fresh)} fresh, {len(stale)} stale, {len(missing)} missing ===")
    for s in stale:
        print(f"  - {s}")

    sys.exit(1 if stale or missing else 0)


if __name__ == "__main__":
    main()
