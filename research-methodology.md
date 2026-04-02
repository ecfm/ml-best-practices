# Research Methodology Practices

Lessons for conducting research with AI agents — literature review, critical analysis,
and research direction planning. Complements experiment-practices.md (running experiments)
and pipeline-architecture.md (code architecture).

---

## 1. Agent Instruction Iteration

When designing agent tasks (lit review, paper extraction, analysis), iterate the
instructions before scaling:

1. **Design** the schema/instructions
2. **Test on 1 example** where you already know the answer
3. **Compare** agent output to ground truth — identify systematic errors
4. **Fix instructions** to address each error type
5. **Re-test** on the same example until clean
6. **Test on 2-3 diverse examples** (different paper types, domains)
7. **Then batch** the remaining work

Common instruction failure modes:
- Agent guesses when it should say "not_reported"
- Agent interpolates ranges instead of listing exact values
- Agent confuses what the paper does vs what the paper studies
- Agent defaults to a value instead of searching the paper text
- Temporal reasoning errors (paper A can't build on paper B if B is newer)

Each failure mode becomes an explicit rule in the instructions. The instructions
grow by accumulating specific fixes — not by adding general advice.

**Anti-pattern:** Writing perfect-sounding instructions once and launching 30 agents.
The first agent reveals 5 systematic errors, but you've already wasted 29 runs.

## 2. Structured Paper Extraction: Link Claims to Experiments

When extracting technical details from papers, the fundamental unit is the
**experiment**, not the paper. A paper is a collection of claims, each supported
by specific experiments.

Structure:
```
claims:
  - claim: "X achieves Y"
    evidence: [experiment_ids]
    caveats: "only tested on Z"

experiments:
  - id: "e1"
    models: [exact names]
    intervention: {type, target, layers, hyperparams}
    dataset: {name, origin, size}
    evaluation: {method, judge, criteria}
    result: "specific number"
    figure: "Figure 4a"
```

This prevents the common failure of citing a paper's claim without knowing:
- Which models it was tested on (generalizability)
- What evaluation method was used (teacher-forcing inflation?)
- How many seeds/runs (reliability)
- What the exact decision criteria were (reproducibility)

A reusable YAML schema and agent instructions for this are in
[templates/paper-cards/](templates/paper-cards/).

## 3. External Adversarial Review

Before committing to a research direction, send your analysis to a different
model/perspective for critique. Use Codex MCP with GPT-5.4:

```
mcp__codex__codex with model="gpt-5.4"
```

### What to include in the prompt
- Full context from the lit review (key findings with arXiv IDs and evidential status)
- The specific proposal/analysis being critiqued
- Self-identified weaknesses (shows good faith, gets more targeted critique)
- Explicit questions that invite pushback

### Prompt patterns that work
- "Does experiment X actually provide evidence for hypothesis Y, or is it confounded by Z?"
- "Is my hierarchy ordering defensible? Which levels am I overconfident about?"
- "What experimental controls would make this decisive vs hand-wavy?"
- "What alternative directions would YOU suggest given this landscape?"

### Key lesson: sufficiency ≠ mediation
Showing that X produces Y (sufficiency) is NOT the same as showing that
interventions cause Y *through* X (mediation). Mediation requires:
1. Show T increases M (intervention activates candidate mediator)
2. Block M during T, show harmful effect disappears (necessity)
3. Show intended effect of T survives blocking M (specificity)
4. Apply M alone, show it partially recapitulates the harm (sufficiency)
5. Repeat across intervention types (generality)

This distinction is the #1 thing external critique catches that self-review
misses.

## 4. Evidential Hierarchy

When building arguments across papers, classify each finding by evidential
strength before using it as a premise:

| Level | Meaning | Trust |
|-------|---------|-------|
| Analytically proven | Formal mathematical proof | High — check assumptions |
| Demonstrated empirically | Tested across models/datasets with controls | Medium-high — check generalizability |
| Suggested | Single-setting observation or limited testing | Medium — needs replication |
| Proposed | Theoretical framework without validation | Low — treat as hypothesis |

Do not build load-bearing arguments on "suggested" findings. If your core
claim depends on a single-setting preprint, flag this explicitly.

Also track:
- **Peer review status**: Published (ICML, NeurIPS) > preprint > withdrawn
- **Replication**: Tested on N model families > single model
- **Evaluation method**: Open-ended generation > teacher-forcing (the latter
  inflates results by up to 2.5x in model editing)

## 5. Cross-Paper Contradiction Detection

When synthesizing across papers, explicitly check for contradictions. Common
types:

- **Scope mismatch**: Paper A shows X at one layer, paper B shows ¬X at all
  layers. Not a real contradiction — different scope.
- **Metric mismatch**: Paper A measures success on direct recall, paper B on
  downstream consequences. Both can be "right" simultaneously.
- **Level mismatch**: Paper A works in weight space, paper B in activation space.
  These can give different answers because they're different projections of the
  same phenomenon.
- **Genuine contradiction**: Paper A and B measure the same thing on the same
  level and disagree. Resolve by checking sample size, evaluation method,
  and model family.

The most productive contradictions are ones where two papers make the SAME
measurement but get different results — these reveal hidden confounds or
boundary conditions.

## 6. Cross-Literature Analysis Structure

When synthesizing across a large body of papers, use focused disputes rather
than grand narratives:

### Identify specific disputes
Find points where the literature actually disagrees. For each dispute:
- What's established (not in dispute)
- Competing explanations with best evidence for each
- What evidence would resolve it
- Current assessment

Grand "worldviews" (e.g., "it's all superposition" vs "it's all optimization")
create strawmen. Specific disputes (e.g., "is norm growth cause or symptom of
editing failure?") produce testable predictions.

### Build explanatory hierarchy
Classify findings by depth, but constrain by evidential strength:
- A weakly-evidenced finding should NOT anchor the hierarchy just because it
  sounds fundamental
- Each level should predict: "if this is the bottleneck, then intervention X
  should help and Y should not"
- Without falsifiable predictions, the hierarchy is taxonomy, not theory

### Check "unifications" rigorously
"Three views of one object" claims (e.g., weight-space subspace = activation-space
direction = loss-landscape attractor) require cross-space intervention transfer
to confirm. Co-occurrence is suggestive, not conclusive.

### Map contradictions carefully
Most apparent contradictions are scope/metric/level mismatches:
- Scope: Paper A tested layer 24, Paper B tested all layers
- Metric: Paper A measured direct recall, Paper B measured downstream consequences
- Level: Paper A analyzed weights, Paper B analyzed activations

Only flag as genuine when two papers measure the same thing at the same level
and disagree.

## 7. Separate Description from Explanation

When reviewing a field, keep three levels distinct:

1. **What happens** (established facts): "Sequential editing causes norm growth"
2. **How it happens** (mechanisms): "The L&E update rule has exponential growth"
3. **Why it happens** (root causes): "Feature superposition makes perturbation non-local"

Most papers establish level 1 (empirical observation), some establish level 2
(specific mechanism), and very few establish level 3 (root cause). Do not
upgrade a level-1 finding to level-3 status because it sounds like an
explanation.

Test: could the finding be a symptom of something deeper? If yes, it's not a
root cause. "Norm growth causes editing collapse" could be true, but norm
growth might itself be a symptom of a deeper geometric property. Classify
accordingly.

## 8. Paper Verification Review Process

Multi-layer review system for verifying empirical claims in research papers
before submission. Learned from Validation Dilemma paper review (2026-04).

### Phase 0: Automated checks (no LLM, run first)

Two scripts that catch the most common errors deterministically:

1. **Number reconciliation**: Parse provenance/claims mapping → grep .tex for
   claimed values → read source CSVs/JSONs → flag mismatches. Catches stale
   numbers after edit rounds (e.g., script rerun changed 88.4% to 87.8% but
   paper wasn't updated).

2. **Staleness detection**: Compare script modification times and hashes against
   output files. If script changed but output didn't → stale. Record baseline
   hashes after each verified rerun.

These replace LLM-based consistency checks (C-tracks) which are slower, more
expensive, and less reliable for mechanical matching.

### Phase 1: LLM review — split by cognitive mode, not by file

Per-file splits (c1=script_A.py, c2=script_B.py) cause 3x duplication of
cross-cutting issues. Instead, split by the *type of thinking* required:

| Layer | Tracks | Cognitive mode | Best model |
|-------|--------|---------------|------------|
| Execute | E1-E4 | Run script, compare output to claimed values | Codex (needs tools) |
| Methodology | M1-M3 | "Does this metric measure what you think?" | Opus (deep reasoning) |
| Adversarial | A1-A3 | "Can I break this argument?" | Opus (creative) |
| Consistency | C1-C2 | Replaced by Phase 0 scripts | Python (deterministic) |

Key design rules:
- **Known-issues injection**: Prepend resolved issues to M/A prompts so reviewers
  don't rediscover them. Auto-extract from DECISIONS.md.
- **Execution tracks mandate running**: "Run it if helpful" → "You MUST run this
  script." The best findings came from actual execution.
- **Adversarial tracks give a thesis to attack**: "The paper claims X. Try to
  construct a counterexample" beats "review this for issues."
- **Methodology tracks separate "is the formula correct?" from "is the metric
  meaningful?"** — reviewers default to formula-checking; metric-questioning must
  be explicitly prompted.

### Phase 2: High-value specialized passes

- **Naive reader**: Someone unfamiliar reads intro + framework cold, reports
  confusion points. The #1 cause of journal rejection is "I didn't understand
  the contribution."
- **Citation verification**: For each \cite{}, check that the attributed claim
  matches the cited paper's actual content. Methodological claims ("X used Y
  approach") are highest risk.
- **Missing analyses** (reviewer simulation): "You are a reviewer. What analyses
  are absent that you would demand?" — then implement them. This is offense, not
  defense. Highest-ROI track in practice.

### Triage: not every gap needs addressing

Adversarial tracks will flag "missing analyses." Before implementing, check:
1. **Does the data support it?** If the dataset structure doesn't allow the analysis, skip.
2. **Would reviewers independently notice?** If the gap requires specialized knowledge to spot, most reviewers won't raise it.
3. **Does acknowledging it strengthen the paper?** Adding a limitation that the data can't address just invites scrutiny.

Only preempt gaps that reviewers will catch and that weaken the argument if unaddressed. Don't volunteer weaknesses nobody would find.

**Document pushbacks, not just decisions.** For every rejected finding, record:
- What was flagged
- Why it was rejected (which of the 3 questions above failed)
- Why it won't resurface (e.g., "already addressed at line X", "data doesn't support")

Without recorded pushbacks, the next round's reviewer (or a real referee) may
rediscover the same point and you'll re-triage from scratch.

### Phase 3: Post-fix verification

Re-run Phase 0 after every fix round. Confirm 0 mismatches before submission.
Phase 0 catches stale provenance entries and counts that drift after reruns.

### Multi-model convergence

Use a **different model family** for the re-review round (e.g., Codex/Opus for
Round 2, Gemini for Round 3). Different models catch different things — same-model
re-review has blind spots. Convergence is established when a fresh model family
finds 0 new must-fix items.

### Setup prerequisites

Before the first review round, ensure these are in place:

1. **Provenance mapping** (`provenance.yaml` or equivalent): every empirical number
   in the paper traces to `source_file` → `script` → `claim_in_tex`. Phase 0 reads
   this mapping — without it, number reconciliation can't run.

2. **Active constraints file** (`DECISIONS.md` or equivalent): living document of
   analytical constraints (things that must stay true) and open issues. Constraints
   prevent future rounds from re-introducing fixed bugs. Updated after each round;
   resolved items move to a "Resolved" section with date and round reference.

3. **Pre-commit hooks**: warn when `.tex` files change without updating provenance.
   Catches the most common post-fix error (update the number, forget the mapping).

### Infrastructure lessons

- **Bash is too fragile for orchestration.** `wait -n` with PID arrays, `set -e`
  with `wait`, and codex `-o` flag all broke. Use Python subprocess or GNU
  parallel.
- **Heavy tracks need splitting upfront.** If a track reads 5+ files or all
  .tex + all CSVs, split it. Estimate: 1-2 files per track for E/C, 3-4 for M,
  1-2 for A.
- **Rate limits kill batch runs.** Build fallback to alternative models (Codex →
  Claude Opus → Gemini) into the orchestrator.
- **Track outputs to files, not stdout.** Some LLM CLI tools have unreliable
  `-o` flags. Always capture stdout as backup.
- **Smoke-test before full batch.** Run 1-2 tracks first to verify prompts work,
  file access is correct, and output format is usable. Then launch the full batch.

## 9. Claims Tracking System

Complements provenance (data lineage) with argumentative structure tracking.
Learned from Validation Dilemma paper (2026-04).

### Boundary with provenance

Provenance tracks **numbers → data files + scripts**. Claims track **assertions →
evidence bundles (provenance refs, citations, other claims, logical premises)**.
Claims reference provenance via `provenance_id` and `provenance_claim` substring
matching — they never duplicate `source:`, `script:`, or `sha256:` fields.

### Claim schema

```yaml
- id: q2_baselines_dominate
  text: "Ridge on 4 demographics outperforms GPT-4 on every decision metric"
  type: empirical    # empirical | logical | literature | interpretive | scope | qualification
  strength: strong   # strong | moderate | weak | contested
  locations:
    - file: main.tex
      quote: "exact substring, 20-80 chars"
  evidence:
    - provenance_id: tbl_decision_metrics     # → provenance/results/<id>/
    - provenance_claim: "Ridge 47.2%"         # substring of provenance.yaml claim
    - cite: krsteski2025                      # references.bib key
    - claim_id: q2_individual_weak            # another claim (logical premise)
  supports: [q2_main]        # argument DAG edges
  depends_on: [q2_individual_weak]
  falsified_by: "Condition 2"
```

### What to automate vs what needs agents

**Phase 0 scripts (deterministic, run first):**
- Quote verification: grep each `quote` in .tex files
- Provenance ref validation: check provenance_id dirs exist, provenance_claim
  substrings match, cite keys in bib, LaTeX labels exist
- Cross-location hedge consistency: multi-location claims carry matching qualifiers
- Staleness: git-diff proximity + upstream chain from `check_staleness.py`
- Orphan detection: provenance entries not referenced by any claim
- Evidence graph generation: DAG from supports/depends_on edges

**LLM agents (judgment, run after Phase 0):**
- Claim extraction from .tex sections
- Evidence linking and strength rating
- Load-bearing analysis (what collapses if a claim is falsified)

### Staleness chain

```
L0: Script changed  →  L1: Output stale  →  L2: Provenance stale  →  L3: Claim stale
(check_staleness.py)   (check_staleness.py)  (check_claims.py)       (check_claims.py)
```

Track verified-at commit per claims file. `--mark-verified` advances to HEAD.
Three signals: quote-gone (grep fails), git-diff proximity (±5 lines of changed
hunk), upstream chain propagation.

### Key design decisions

- **One YAML file per argument section**, not per claim. Matches paper structure,
  manageable file count (~12 for a full paper).
- **Per-file `_meta.verified_at_commit`**, not per-claim timestamps. Low overhead,
  sufficient granularity — staleness check flags individual claims within stale files.
- **Hedge consistency checker** with 200-char context window around each quote.
  Catches scope drift between abstract and body. Keep the hedge phrase list small
  and paper-specific.
- **`VERIFIED_OK` list in check_numbers.py** for mismatches that are checker
  limitations (boolean column aggregates, method-filtered counts, sign-flipped CIs,
  metadata numbers). Documents the manual triage so it doesn't repeat.
- **Extraction agents need calibration.** Extract one section manually as ground
  truth, test the agent prompt against it, fix systematic errors, then batch.
  Common agent errors: en-dash vs hyphen in provenance_claim substrings, case
  sensitivity in quotes, confusing method descriptions for claims.
