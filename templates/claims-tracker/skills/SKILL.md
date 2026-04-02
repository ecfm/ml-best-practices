---
name: paper-claims
description: Validate, extract, and verify manuscript claims against evidence. Use when editing .tex files, before submission, or when asked about claims/evidence/provenance.
allowed-tools: Read Grep Glob Bash Agent
effort: high
argument-hint: "[check | extract <section> | verify]"
---

# Paper Claims

Manages the claims tracking system for a LaTeX manuscript. Three modes based on `$ARGUMENTS`:

## Mode: check (default)

Run when `$ARGUMENTS` is empty or "check".

Run the full Phase 0 validation suite:

```
python scripts/check_all.py
```

If the project doesn't have check_all.py, run individual scripts:
```
python scripts/check_claims.py --staleness --orphans
```

Report issues by category:
- **STALE**: quotes no longer in .tex — update the quote in claims YAML
- **SUSPECT**: nearby text changed — check if claim still holds
- **Warnings**: hedge inconsistencies, unresolved refs — fix claims YAML or .tex
- **Orphaned provenance**: unlinked entries — add provenance_claim refs to claims

## Mode: extract

Run when `$ARGUMENTS` starts with "extract".

1. Identify the .tex file and line range for the requested section
2. Read provenance.yaml (or equivalent) for evidence references
3. Read existing claims files for cross-reference IDs
4. Extract claims to `claims/<main|si>/<section>.yaml`

### Claim schema

```yaml
_meta:
  verified_at_commit: "<current HEAD short hash>"
  verified_at_date: "<today>"
  tex_source: "<tex filename>"
  tex_sections: ["<section names>"]

claims:
  - id: <prefix>_<name>       # section prefix + descriptive name
    text: "one sentence paraphrase"
    type: <empirical|logical|literature|interpretive|scope|qualification>
    strength: <strong|moderate|weak|contested>
    locations:
      - file: <tex file>
        quote: "exact 20-80 char substring"
    evidence:
      - provenance_claim: "substring of provenance entry"  # data lineage
      - cite: <bibtex key>                                 # bibliography
      - claim_id: <other claim ID>                         # logical premise
      - figure: <label>                                    # LaTeX label
      - table: <label>
    supports: [<parent claim IDs>]
    depends_on: [<premise claim IDs>]
    qualifications: ["caveats in text"]
```

### What is a claim
An assertion a reader could ask "what's the evidence?" — NOT method descriptions, definitions, or transitions.

### Strength criteria
- **strong**: own-data with provenance + replicated independently
- **moderate**: own-data single dataset, or strong literature
- **weak**: argument from absence, interpretive, sparse
- **contested**: evidence for and against

After writing, validate: `python scripts/check_claims.py`

## Mode: verify

Run when `$ARGUMENTS` is "verify".

1. `python scripts/check_claims.py --staleness` — find stale/suspect claims
2. Review and fix each flagged claim
3. `python scripts/check_claims.py --mark-verified` — advance verified_at_commit to HEAD
4. `python scripts/generate_evidence_graph.py && python scripts/generate_claim_index.py`
5. Report: total claims, edges, longest chain, weak load-bearing claims

## Boundary with provenance

Claims reference provenance entries — they never duplicate `source:`, `script:`, or `sha256:` fields. Use `provenance_claim` (substring match) and `provenance_id` (directory name) to link.
