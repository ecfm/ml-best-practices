#!/usr/bin/env python3
"""Claims validation and staleness detection.

Template from claude-toolkit/templates/claims-tracker/
Customize the CONFIG dict below for your project.

Validates claim YAML files against the manuscript and provenance system.

Usage:
    python scripts/check_claims.py              # full validation
    python scripts/check_claims.py --staleness   # + git diff proximity checks
    python scripts/check_claims.py --orphans     # provenance entries not referenced by any claim
    python scripts/check_claims.py --mark-verified  # update _meta.verified_at_commit to HEAD

Exit codes: 0 = all OK, 1 = issues found, 2 = errors
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: uv pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ── Project configuration ──
# Customize these for your project.

CONFIG = {
    "tex_files": None,  # auto-discovered from claims _meta.tex_source; override to restrict
    "hedge_phrases": [],  # project-specific scoping language to check for consistency
    "claims_dir": "claims",
    "provenance_yaml": "provenance.yaml",
    "provenance_results_dir": "provenance/results",
    "references_bib": "references.bib",
}

# ── Paths (derived from CONFIG) ──

PAPER = Path(__file__).resolve().parent.parent
CLAIMS_DIR = PAPER / CONFIG["claims_dir"]
PROVENANCE_YAML = PAPER / CONFIG["provenance_yaml"]
PROVENANCE_RESULTS = PAPER / CONFIG["provenance_results_dir"]
REFERENCES_BIB = PAPER / CONFIG["references_bib"]

VALID_TYPES = {"empirical", "logical", "literature", "interpretive", "scope", "qualification"}
VALID_STRENGTHS = {"strong", "moderate", "weak", "contested"}
EVIDENCE_REF_KEYS = {"provenance_id", "provenance_claim", "cite", "claim_id", "figure", "table"}


# ── Auto-discovery ──


def discover_tex_files(claims_files):
    """Discover valid .tex files from _meta.tex_source across all claims files.

    If CONFIG["tex_files"] is set, use that instead.
    """
    if CONFIG["tex_files"] is not None:
        return set(CONFIG["tex_files"])

    tex_files = set()
    for path in claims_files:
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                meta = data.get("_meta", {})
                src = meta.get("tex_source")
                if src:
                    tex_files.add(src)
        except Exception:
            pass

    if not tex_files:
        print("WARN: no tex_source found in any claims file _meta", file=sys.stderr)
    return tex_files


# ── Helpers ──


def load_claims_file(path):
    """Load a claims YAML file, return (meta, claims) or raise."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected dict at top level")
    meta = data.get("_meta", {})
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError(f"{path.name}: 'claims' must be a list")
    return meta, claims


def find_all_claims_files():
    """Find all .yaml files in claims/ subdirectories."""
    files = []
    if not CLAIMS_DIR.exists():
        return files
    for subdir in sorted(d.name for d in CLAIMS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")):
        d = CLAIMS_DIR / subdir
        files.extend(sorted(d.glob("*.yaml")))
    return files


def load_all_claim_ids(claims_files):
    """Load all claim IDs across all files."""
    all_ids = {}
    for path in claims_files:
        try:
            _, claims = load_claims_file(path)
            for c in claims:
                cid = c.get("id", "MISSING")
                if cid in all_ids:
                    all_ids[cid].append(path.name)
                else:
                    all_ids[cid] = [path.name]
        except Exception:
            pass
    return all_ids


def load_tex(filename):
    """Load a .tex file's content."""
    path = PAPER / filename
    if not path.exists():
        return None
    with open(path) as f:
        return f.read()


def load_tex_lines(filename):
    """Load .tex file as list of (line_number, line_text)."""
    path = PAPER / filename
    if not path.exists():
        return []
    with open(path) as f:
        return list(enumerate(f.readlines(), 1))


def find_quote_line(tex_lines, quote):
    """Find line number where quote appears. Returns (line_no, True) or (None, False)."""
    for line_no, line_text in tex_lines:
        if quote in line_text:
            return line_no, True
    return None, False


def load_provenance_claims():
    """Load all 'claim' fields from provenance.yaml."""
    if not PROVENANCE_YAML.exists():
        return []
    claims = []
    with open(PROVENANCE_YAML) as f:
        current_claim = None
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- claim:"):
                current_claim = stripped.split(":", 1)[1].strip().strip('"')
                claims.append(current_claim)
    return claims


def load_provenance_result_ids():
    """List all result IDs in provenance/results/."""
    if not PROVENANCE_RESULTS.exists():
        return set()
    return {d.name for d in PROVENANCE_RESULTS.iterdir() if d.is_dir()}


def load_bib_keys():
    """Extract all bibtex keys from references.bib."""
    if not REFERENCES_BIB.exists():
        return set()
    with open(REFERENCES_BIB) as f:
        content = f.read()
    return set(re.findall(r'@\w+\{([^,]+),', content))


def load_tex_labels(valid_tex_files):
    """Extract all \\label{} from .tex files."""
    labels = set()
    for filename in valid_tex_files:
        tex = load_tex(filename)
        if tex:
            labels.update(re.findall(r'\\label\{([^}]+)\}', tex))
    return labels


def git_head_commit():
    """Get current HEAD short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PAPER
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_diff_changed_lines(base_commit, tex_file):
    """Get line ranges changed between base_commit and HEAD for a .tex file.

    Returns list of (start_line, end_line) tuples in the NEW file.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_commit}..HEAD", "--unified=0", "--", tex_file],
            capture_output=True, text=True, cwd=PAPER
        )
    except Exception:
        return []

    ranges = []
    for line in result.stdout.splitlines():
        # Parse @@ -old_start,old_count +new_start,new_count @@
        m = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) else 1
            if count > 0:
                ranges.append((start, start + count - 1))
    return ranges


def line_in_changed_region(line_no, changed_ranges, proximity=5):
    """Check if a line number falls within proximity of any changed range."""
    if line_no is None:
        return False
    for start, end in changed_ranges:
        if start - proximity <= line_no <= end + proximity:
            return True
    return False


# ── Validation checks ──


def validate_schema(meta, claims, filename, all_ids, valid_tex_files):
    """Check required fields, valid enums, valid references."""
    issues = []

    # Meta checks
    if not meta.get("verified_at_commit"):
        issues.append(("ERROR", filename, "_meta", "missing verified_at_commit"))
    if not meta.get("tex_source"):
        issues.append(("ERROR", filename, "_meta", "missing tex_source"))
    elif meta["tex_source"] not in valid_tex_files:
        issues.append(("WARN", filename, "_meta", f"tex_source '{meta['tex_source']}' not in {valid_tex_files}"))

    for c in claims:
        cid = c.get("id", "MISSING_ID")

        # Required fields
        for field in ["id", "text", "type", "locations"]:
            if field not in c:
                issues.append(("ERROR", filename, cid, f"missing required field '{field}'"))

        # Type enum
        if c.get("type") and c["type"] not in VALID_TYPES:
            issues.append(("ERROR", filename, cid, f"invalid type '{c['type']}'"))

        # Strength enum
        if c.get("strength") and c["strength"] not in VALID_STRENGTHS:
            issues.append(("ERROR", filename, cid, f"invalid strength '{c['strength']}'"))

        # Location entries
        for loc in c.get("locations", []):
            if "file" not in loc:
                issues.append(("ERROR", filename, cid, "location missing 'file'"))
            elif loc["file"] not in valid_tex_files:
                issues.append(("WARN", filename, cid, f"location file '{loc['file']}' not in {valid_tex_files}"))
            if "quote" not in loc:
                issues.append(("ERROR", filename, cid, "location missing 'quote'"))

        # Evidence entries
        for ev in c.get("evidence", []):
            if not isinstance(ev, dict):
                issues.append(("ERROR", filename, cid, f"evidence entry is not a dict: {ev}"))
                continue
            ref_keys = set(ev.keys()) & EVIDENCE_REF_KEYS
            if not ref_keys:
                issues.append(("WARN", filename, cid, f"evidence entry has no reference key: {ev}"))

        # Cross-references: supports, depends_on, replicated_by
        for ref_field in ["supports", "depends_on", "replicated_by"]:
            for ref_id in c.get(ref_field, []):
                if ref_id not in all_ids:
                    issues.append(("WARN", filename, cid, f"{ref_field} references unknown claim '{ref_id}'"))

        # Duplicate IDs
        if cid in all_ids and len(all_ids[cid]) > 1:
            issues.append(("ERROR", filename, cid, f"duplicate ID found in: {all_ids[cid]}"))

    return issues


def verify_locations(claims, filename, tex_cache):
    """Check that each quote exists in its .tex file."""
    issues = []
    for c in claims:
        cid = c.get("id", "MISSING_ID")
        for loc in c.get("locations", []):
            tex_file = loc.get("file")
            quote = loc.get("quote")
            if not tex_file or not quote:
                continue
            tex = tex_cache.get(tex_file)
            if tex is None:
                issues.append(("ERROR", filename, cid, f"tex file '{tex_file}' not found"))
                continue
            if quote not in tex:
                issues.append(("STALE", filename, cid, f"quote not found in {tex_file}: \"{quote[:60]}...\""))
    return issues


def verify_evidence_refs(claims, filename, provenance_claims, provenance_ids, bib_keys, tex_labels):
    """Check that evidence references resolve."""
    issues = []
    for c in claims:
        cid = c.get("id", "MISSING_ID")
        for ev in c.get("evidence", []):
            if not isinstance(ev, dict):
                continue

            if "provenance_id" in ev:
                pid = ev["provenance_id"]
                if pid not in provenance_ids:
                    issues.append(("ERROR", filename, cid, f"provenance_id '{pid}' not in provenance/results/"))

            if "provenance_claim" in ev:
                pc = ev["provenance_claim"]
                if not any(pc in claim_text for claim_text in provenance_claims):
                    issues.append(("WARN", filename, cid, f"provenance_claim not found as substring: \"{pc[:50]}\""))

            if "cite" in ev:
                key = ev["cite"]
                if key not in bib_keys:
                    issues.append(("ERROR", filename, cid, f"cite key '{key}' not in references.bib"))

            if "figure" in ev:
                label = ev["figure"]
                if label not in tex_labels:
                    issues.append(("WARN", filename, cid, f"figure label '{label}' not in .tex files"))

            if "table" in ev:
                label = ev["table"]
                if label not in tex_labels:
                    issues.append(("WARN", filename, cid, f"table label '{label}' not in .tex files"))

    return issues


def check_staleness_git(meta, claims, filename, tex_lines_cache):
    """Check which claims fall in git-changed regions since last verification."""
    issues = []
    base = meta.get("verified_at_commit")
    tex_source = meta.get("tex_source")
    if not base or not tex_source:
        return issues

    changed_ranges = git_diff_changed_lines(base, tex_source)
    if not changed_ranges:
        return issues

    tex_lines = tex_lines_cache.get(tex_source, [])

    for c in claims:
        cid = c.get("id", "MISSING_ID")
        for loc in c.get("locations", []):
            if loc.get("file") != tex_source:
                continue
            quote = loc.get("quote", "")
            line_no, found = find_quote_line(tex_lines, quote)
            if found and line_in_changed_region(line_no, changed_ranges):
                issues.append(("SUSPECT", filename, cid,
                               f"quote at line {line_no} is near changed region in {tex_source}"))

    return issues


def find_orphaned_provenance(all_evidence_refs, provenance_claims):
    """Find provenance.yaml entries not referenced by any claim."""
    orphans = []
    for pc in provenance_claims:
        referenced = False
        for ref_set in all_evidence_refs:
            for ref in ref_set:
                if isinstance(ref, dict) and "provenance_claim" in ref:
                    if ref["provenance_claim"] in pc:
                        referenced = True
                        break
                if isinstance(ref, dict) and "provenance_id" in ref:
                    # provenance_id references are matched separately
                    pass
            if referenced:
                break
        if not referenced:
            orphans.append(pc)
    return orphans


def check_cross_location_consistency(claims, filename, tex_cache):
    """Check that multi-location claims carry consistent hedges.

    Uses CONFIG["hedge_phrases"] — if empty, this check is skipped.
    """
    hedge_phrases = CONFIG["hedge_phrases"]
    if not hedge_phrases:
        return []

    issues = []

    for c in claims:
        cid = c.get("id", "MISSING_ID")
        locs = c.get("locations", [])
        if len(locs) < 2:
            continue

        # For each hedge, check if it appears near each location
        for hedge in hedge_phrases:
            contexts = []
            for loc in locs:
                tex = tex_cache.get(loc.get("file"))
                quote = loc.get("quote", "")
                if not tex or not quote:
                    continue
                idx = tex.find(quote)
                if idx == -1:
                    continue
                # Get surrounding context (200 chars before and after)
                start = max(0, idx - 200)
                end = min(len(tex), idx + len(quote) + 200)
                context = tex[start:end]
                contexts.append(hedge.lower() in context.lower())

            if contexts and any(contexts) and not all(contexts):
                has = [i for i, v in enumerate(contexts) if v]
                missing = [i for i, v in enumerate(contexts) if not v]
                issues.append(("WARN", filename, cid,
                               f"hedge '{hedge}' present at location(s) {has} but absent at {missing}"))

    return issues


# ── Mark verified ──


def mark_verified(claims_files):
    """Update _meta.verified_at_commit to HEAD for all claims files."""
    head = git_head_commit()
    if not head:
        print("ERROR: could not determine HEAD commit", file=sys.stderr)
        return False

    from datetime import date
    today = date.today().isoformat()

    updated = 0
    for path in claims_files:
        with open(path) as f:
            content = f.read()

        # Replace verified_at_commit (handles both single and double quotes, or unquoted)
        new_content = re.sub(
            r'(verified_at_commit:\s*)["\']?[^"\'"\n]+["\']?',
            f'\\1"{head}"',
            content
        )
        new_content = re.sub(
            r'(verified_at_date:\s*)["\']?[^"\'"\n]+["\']?',
            f'\\1"{today}"',
            new_content
        )

        if new_content != content:
            with open(path, "w") as f:
                f.write(new_content)
            updated += 1
            print(f"  Updated {path.name}: verified_at_commit={head}, date={today}")

    print(f"\n{updated} file(s) updated to commit {head}")
    return True


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Claims validation and staleness detection")
    parser.add_argument("--staleness", action="store_true", help="Include staleness checks")
    parser.add_argument("--orphans", action="store_true", help="Find orphaned provenance entries")
    parser.add_argument("--mark-verified", action="store_true", help="Update verified_at_commit to HEAD")
    parser.add_argument("--brief", action="store_true", help="Summary only")
    args = parser.parse_args()

    claims_files = find_all_claims_files()
    if not claims_files:
        print(f"No claims files found in {CLAIMS_DIR}/")
        sys.exit(2)

    if args.mark_verified:
        mark_verified(claims_files)
        sys.exit(0)

    # Auto-discover .tex files from claims metadata
    valid_tex_files = discover_tex_files(claims_files)

    # Load reference data
    all_ids = load_all_claim_ids(claims_files)
    provenance_claims = load_provenance_claims()
    provenance_ids = load_provenance_result_ids()
    bib_keys = load_bib_keys()
    tex_labels = load_tex_labels(valid_tex_files)
    tex_cache = {f: load_tex(f) for f in valid_tex_files}
    tex_lines_cache = {f: load_tex_lines(f) for f in valid_tex_files}

    all_issues = []
    total_claims = 0
    all_evidence_refs = []

    for path in claims_files:
        try:
            meta, claims = load_claims_file(path)
        except Exception as e:
            all_issues.append(("ERROR", path.name, "-", f"failed to load: {e}"))
            continue

        total_claims += len(claims)

        # Collect all evidence refs for orphan check
        for c in claims:
            all_evidence_refs.append(c.get("evidence", []))

        # Schema validation
        all_issues.extend(validate_schema(meta, claims, path.name, all_ids, valid_tex_files))

        # Location verification
        all_issues.extend(verify_locations(claims, path.name, tex_cache))

        # Evidence reference verification
        all_issues.extend(verify_evidence_refs(claims, path.name, provenance_claims, provenance_ids, bib_keys, tex_labels))

        # Cross-location consistency
        all_issues.extend(check_cross_location_consistency(claims, path.name, tex_cache))

        # Staleness checks
        if args.staleness:
            all_issues.extend(check_staleness_git(meta, claims, path.name, tex_lines_cache))

    # Orphan detection
    if args.orphans:
        orphans = find_orphaned_provenance(all_evidence_refs, provenance_claims)
        for o in orphans:
            all_issues.append(("ORPHAN", "provenance.yaml", "-", f"unreferenced: \"{o[:70]}\""))

    # ── Report ──

    errors = [i for i in all_issues if i[0] == "ERROR"]
    warnings = [i for i in all_issues if i[0] == "WARN"]
    stale = [i for i in all_issues if i[0] == "STALE"]
    suspect = [i for i in all_issues if i[0] == "SUSPECT"]
    orphan_issues = [i for i in all_issues if i[0] == "ORPHAN"]

    print(f"=== Claims Validation ===")
    print(f"Files: {len(claims_files)}")
    print(f"Claims: {total_claims}")
    print(f"Unique IDs: {len(all_ids)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Stale: {len(stale)}")
    if args.staleness:
        print(f"Suspect: {len(suspect)}")
    if args.orphans:
        print(f"Orphaned provenance: {len(orphan_issues)}")

    if not args.brief:
        for severity in ["ERROR", "STALE", "SUSPECT", "WARN", "ORPHAN"]:
            items = [i for i in all_issues if i[0] == severity]
            if items:
                print(f"\n--- {severity} ({len(items)}) ---")
                for sev, file, cid, msg in items:
                    print(f"  [{file}] {cid}: {msg}")

    has_issues = bool(errors or stale)
    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
