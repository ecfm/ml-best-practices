#!/usr/bin/env python3
"""Generate a human-readable claim index from claims files.

Template from claude-toolkit/templates/claims-tracker/
Customize the CONFIG dict below for your project.

Reads all claims/*.yaml and outputs claims/claim_index.md -- a summary table
with ID, one-line text, type, strength, evidence count, and location count.

Usage:
    python scripts/generate_claim_index.py
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: uv pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ── Project configuration ──

CONFIG = {
    "claims_dir": "claims",
    # Section ordering for sorting claims by position in the paper.
    # Map section prefix (from claim ID before first '_') to a sort key.
    # Lower values sort earlier. Customize for your paper structure.
    "section_order": {},
}

PAPER = Path(__file__).resolve().parent.parent
CLAIMS_DIR = PAPER / CONFIG["claims_dir"]


def section_key(claim_id):
    """Extract section prefix from claim ID for sorting."""
    prefix = claim_id.split("_")[0]
    return CONFIG["section_order"].get(prefix, 99)


def truncate(text, maxlen=70):
    """Truncate text to maxlen, adding ... if needed."""
    text = text.strip().replace("\n", " ")
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 3] + "..."


def main():
    all_claims = []

    if not CLAIMS_DIR.exists():
        print(f"No claims directory found at {CLAIMS_DIR}/")
        sys.exit(2)

    for subdir in sorted(d.name for d in CLAIMS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")):
        d = CLAIMS_DIR / subdir
        for path in sorted(d.glob("*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            source = f"{subdir}/{path.name}"
            for c in data.get("claims", []):
                all_claims.append({
                    "id": c.get("id", "MISSING"),
                    "text": c.get("text", ""),
                    "type": c.get("type", "?"),
                    "strength": c.get("strength", "-"),
                    "evidence_count": len(c.get("evidence", [])),
                    "location_count": len(c.get("locations", [])),
                    "supports_count": len(c.get("supports", [])),
                    "depends_count": len(c.get("depends_on", [])),
                    "source": source,
                })

    # Sort by argument position
    all_claims.sort(key=lambda c: (section_key(c["id"]), c["id"]))

    # Generate markdown
    lines = [
        "# Claim Index",
        "",
        f"Total claims: {len(all_claims)}",
        "",
        "| ID | Claim | Type | Strength | Ev | Loc | File |",
        "|:---|:------|:-----|:---------|---:|----:|:-----|",
    ]

    type_counts = {}
    strength_counts = {}

    for c in all_claims:
        ctype = c["type"]
        cstrength = c["strength"] if c["strength"] else "-"
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        strength_counts[cstrength] = strength_counts.get(cstrength, 0) + 1

        lines.append(
            f"| `{c['id']}` "
            f"| {truncate(c['text'])} "
            f"| {ctype} "
            f"| {cstrength} "
            f"| {c['evidence_count']} "
            f"| {c['location_count']} "
            f"| {c['source']} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        "**By type:**",
    ])
    for t, n in sorted(type_counts.items()):
        lines.append(f"- {t}: {n}")

    lines.extend([
        "",
        "**By strength:**",
    ])
    for s, n in sorted(strength_counts.items()):
        lines.append(f"- {s}: {n}")

    out_path = CLAIMS_DIR / "claim_index.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"=== Claim Index ===")
    print(f"Claims: {len(all_claims)}")
    print(f"Types: {type_counts}")
    print(f"Strengths: {strength_counts}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
