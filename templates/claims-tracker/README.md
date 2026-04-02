# Claims Tracker Template

A system for tracking, validating, and visualizing the evidence structure of a LaTeX manuscript. Every empirical, logical, and interpretive claim in the paper is extracted into structured YAML, linked to its evidence (provenance entries, citations, figures, tables, other claims), and continuously validated against the manuscript source.

## What it does

- **check_claims.py** -- validates claim YAML files against the manuscript: checks that quotes still exist in .tex files (staleness detection), evidence references resolve (bib keys, provenance IDs, LaTeX labels), schema is valid, cross-references between claims exist, and optionally checks hedge phrase consistency across multi-location claims.
- **generate_evidence_graph.py** -- builds an argument graph from supports/depends_on/replicated_by edges between claims. Outputs YAML (for programmatic use) and DOT (for visualization). Identifies root claims (thesis-level), leaf claims (raw evidence), the longest dependency chain, and weak load-bearing claims.
- **generate_claim_index.py** -- produces a markdown summary table of all claims with type, strength, evidence count, and source file.
- **schema.yaml** -- reference specification for the claims YAML format.
- **skills/SKILL.md** -- Claude Code skill for check/extract/verify workflows.

## Setup

### 1. Copy files into your project

```
your-project/
  scripts/
    check_claims.py
    generate_evidence_graph.py
    generate_claim_index.py
  claims/
    schema.yaml
    main/           # claims from main.tex (or your primary .tex file)
    si/             # claims from supplementary (if applicable)
  .claude/
    skills/
      paper-claims.md   # copy from skills/SKILL.md
```

The scripts expect to live in `scripts/` one level below the project root. They resolve paths as `Path(__file__).resolve().parent.parent`.

### 2. Customize CONFIG

Each script has a `CONFIG` dict at the top. The key settings:

```python
CONFIG = {
    "tex_files": None,          # auto-discovered from claims _meta.tex_source
                                # set to e.g. {"main.tex", "appendix.tex"} to restrict
    "hedge_phrases": [],        # project-specific scoping language, e.g.
                                # ["in our benchmarks", "across the methods we tested"]
    "claims_dir": "claims",     # directory containing claim YAML files
    "provenance_yaml": "provenance.yaml",
    "provenance_results_dir": "provenance/results",
    "references_bib": "references.bib",
}
```

For `generate_claim_index.py`, also customize `section_order` to control how claims sort:

```python
CONFIG = {
    "claims_dir": "claims",
    "section_order": {
        "intro": 0,
        "methods": 1,
        "results": 2,
        "discussion": 3,
    },
}
```

### 3. Create claims directory structure

```
mkdir -p claims/main claims/si
```

Subdirectory names are arbitrary -- the scripts scan all non-hidden subdirectories of `claims/`.

### 4. Install dependency

```
uv pip install pyyaml
```

## Usage workflow

### Extract claims

Use the skill: `/paper-claims extract <section>`

Or manually create YAML files following `schema.yaml`. Each file has a `_meta` block and a `claims` list:

```yaml
_meta:
  verified_at_commit: "abc1234"
  verified_at_date: "2026-04-01"
  tex_source: "main.tex"
  tex_sections: ["introduction"]

claims:
  - id: intro_main_finding
    text: "Our method outperforms baselines on X."
    type: empirical
    strength: moderate
    locations:
      - file: main.tex
        quote: "outperforms all baselines on the X metric"
    evidence:
      - provenance_claim: "X metric comparison"
      - cite: smith2025
      - figure: fig:results
    supports: [disc_conclusion]
```

### Check claims

```bash
python scripts/check_claims.py                # basic validation
python scripts/check_claims.py --staleness     # + git-based staleness detection
python scripts/check_claims.py --orphans       # + orphaned provenance entries
python scripts/check_claims.py --brief         # summary only
```

### Mark verified

After reviewing and fixing all issues:

```bash
python scripts/check_claims.py --mark-verified
```

This updates `_meta.verified_at_commit` to the current HEAD in all claims files.

### Generate outputs

```bash
python scripts/generate_evidence_graph.py      # claims/evidence_graph.yaml + .dot
python scripts/generate_claim_index.py         # claims/claim_index.md
```

To render the DOT graph:

```bash
dot -Tpng claims/evidence_graph.dot -o claims/evidence_graph.png
```

### Full verify cycle (via skill)

`/paper-claims verify` runs the full cycle: check with staleness, fix issues, mark verified, regenerate graph and index.

## How .tex files are discovered

By default, `check_claims.py` scans all claims files and collects the set of `_meta.tex_source` values. This means you don't need to hardcode which .tex files your project uses -- it's inferred from the claims themselves. Override by setting `CONFIG["tex_files"]` to a set of filenames.

## Claim types and strengths

See `schema.yaml` for the full specification. In brief:

**Types**: empirical, logical, literature, interpretive, scope, qualification

**Strengths**: strong (own-data + replicated), moderate (own-data or strong literature), weak (sparse/interpretive), contested (evidence both ways)

## Evidence linking

Claims link to evidence via:

- `provenance_id` -- matches a directory in `provenance/results/<id>/`
- `provenance_claim` -- substring match against `provenance.yaml` claim fields
- `cite` -- bibtex key in `references.bib`
- `claim_id` -- another claim (logical premise)
- `figure` / `table` -- LaTeX `\label{}` references

## Evidence graph analysis

The graph identifies structural properties of your argument:

- **Roots**: thesis-level claims (supported by others, support nothing above them)
- **Leaves**: raw evidence claims (support others but have no supporters)
- **Longest chain**: the critical path through your argument
- **Weak load-bearing**: weak/contested claims that support strong/moderate claims -- these are the most important to strengthen
