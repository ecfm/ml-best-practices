#!/usr/bin/env python3
"""Phase 0: Automated number reconciliation.

Parses a provenance YAML file mapping empirical claims to source files,
then verifies each claimed number appears in the .tex file and matches
the source data.

Configuration:
    Set PAPER_DIR, PROVENANCE_PATH, TEX_FILES, and SOURCE_ROOTS below.

Usage:
    python check_numbers.py              # full check
    python check_numbers.py --brief      # summary only
    python check_numbers.py --fix-list   # print mismatches as fixable items

Exit codes: 0 = all OK, 1 = mismatches found
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ── CONFIGURE THESE ──

PAPER_DIR = Path(__file__).resolve().parent.parent  # adjust to your layout
PROVENANCE_PATH = PAPER_DIR / "provenance.yaml"
TEX_FILES = [PAPER_DIR / "main.tex", PAPER_DIR / "si_appendix.tex"]

# Directories to search for source files (tried in order)
SOURCE_ROOTS = [
    PAPER_DIR.parent,                    # repo root
    PAPER_DIR.parent / "analysis",       # adjust to your analysis dir
    PAPER_DIR.parent / "experiments",    # adjust to your experiments dir
    PAPER_DIR,
]

# Numbers to skip (confidence levels, years, trivial values)
SKIP_VALUES = {0, 1, 2, 3, 90, 95, 99}
SKIP_RANGE = (1990, 2030)  # years

# ── Helpers ──

def load_yaml_entries(path):
    """Parse provenance YAML entries (simplified list parser)."""
    entries = []
    current = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- claim:"):
                if current:
                    entries.append(current)
                current = {"claim": stripped.split(":", 1)[1].strip().strip('"')}
            elif stripped.startswith("file:") and current:
                current["file"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("source:") and current:
                current["source"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("script:") and current:
                current["script"] = stripped.split(":", 1)[1].strip()
    if current:
        entries.append(current)
    return entries


def extract_numbers(text):
    """Extract empirical numeric values, filtering junk."""
    cleaned = re.sub(r'[A-Z][a-z]*\d+[A-Za-z-]*', '', text)  # remove model names
    cleaned = re.sub(r'\d{4}[- ]\d{2}[- ]\d{2}', '', cleaned)  # remove dates
    numbers = []
    for m in re.finditer(r'(?<![A-Za-z])-?\d[\d,]*\.?\d*(?![A-Za-z\d])', cleaned):
        val = m.group().replace(",", "")
        try:
            num = float(val)
        except ValueError:
            continue
        if num in SKIP_VALUES or (SKIP_RANGE[0] <= num <= SKIP_RANGE[1]):
            continue
        numbers.append(num)
    return numbers


def resolve_source(source_str):
    """Resolve source path relative to known roots."""
    source_str = source_str.strip()
    if "(" in source_str or source_str.startswith("inline"):
        return None
    for base in SOURCE_ROOTS:
        candidate = base / source_str
        if candidate.exists():
            return candidate
    return None


def read_source_values(path):
    """Read all numeric values from CSV or JSON."""
    values = {}
    if path.suffix == ".csv":
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    for k, v in row.items():
                        try:
                            values[f"{k}={v}"] = float(v)
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass
    elif path.suffix == ".json":
        try:
            with open(path) as f:
                _extract_json(json.load(f), "", values)
        except Exception:
            pass
    return values


def _extract_json(obj, prefix, values):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _extract_json(v, f"{prefix}.{k}" if prefix else k, values)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _extract_json(v, f"{prefix}[{i}]", values)
    elif isinstance(obj, (int, float)):
        values[prefix] = float(obj)


def find_value_in_source(target, source_values, tolerance=0.05):
    """Match target number in source values (handles pct↔decimal)."""
    matches = []
    for key, val in source_values.items():
        if abs(val - target) < tolerance:
            matches.append((key, val, "exact"))
        elif abs(val * 100 - target) < tolerance:
            matches.append((key, val, "×100"))
        elif target != 0 and abs(val - target / 100) < tolerance / 100:
            matches.append((key, val, "÷100"))
        elif target > 100 and abs(val - target) < 1:
            matches.append((key, val, "integer"))
    return matches


def find_in_tex(number, tex_files):
    """Check if number appears in any .tex file."""
    search_strs = {f"{number:.1f}", f"{number:.2f}"}
    if number == int(number) and number > 1:
        search_strs.add(str(int(number)))
    if 0 < number < 1:
        search_strs.add(f"{number * 100:.1f}")
    for tex_path in tex_files:
        with open(tex_path) as f:
            for i, line in enumerate(f, 1):
                if any(s in line for s in search_strs):
                    return True
    return False


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--fix-list", action="store_true")
    args = parser.parse_args()

    entries = load_yaml_entries(PROVENANCE_PATH)
    verified, mismatches, unresolved = [], [], []

    for entry in entries:
        claim = entry.get("claim", "")
        source_str = entry.get("source", "")
        numbers = extract_numbers(claim)
        if not numbers:
            continue

        source_path = resolve_source(source_str)
        if source_path is None or not source_path.exists():
            unresolved.append({"claim": claim[:80], "source": source_str[:60]})
            continue

        source_values = read_source_values(source_path)

        for num in numbers:
            if num < 0.001 or num > 100000:
                continue
            matches = find_value_in_source(num, source_values)
            in_tex = find_in_tex(num, TEX_FILES)
            if matches:
                verified.append({"number": num, "claim": claim[:60], "in_tex": in_tex})
            else:
                closest = min(source_values.values(), key=lambda v: abs(v - num)) if source_values else None
                mismatches.append({"number": num, "claim": claim[:60], "source": source_str[:50],
                                   "in_tex": in_tex, "closest": closest})

    print(f"=== Number Reconciliation ===")
    print(f"Entries: {len(entries)}, Verified: {len(verified)}, "
          f"Mismatches: {len(mismatches)}, Unresolved: {len(unresolved)}")

    if not args.brief and mismatches:
        print(f"\n--- MISMATCHES ({len(mismatches)}) ---")
        for m in mismatches:
            c = f" (closest: {m['closest']:.4f})" if m['closest'] is not None else ""
            print(f"  {m['number']} in \"{m['claim']}\"{c}")

    if args.fix_list and mismatches:
        print(f"\n--- FIX LIST ---")
        for m in mismatches:
            print(f"- [ ] Verify {m['number']} in \"{m['claim']}\"")

    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
