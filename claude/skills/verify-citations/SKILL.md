---
name: verify-citations
description: Verify claims, quotes, and bib entries in a research document using blind two-step checking. Use when user says "verify citations", "check citations", "fact-check paper", "citation check", or wants to validate references in a LaTeX/markdown document.
argument-hint: <file_path> [--section <name>] [--papers <key1,key2,...>]
allowed-tools: Bash(*), Read, Grep, Glob, Write, Edit, Agent, WebFetch, WebSearch
---

# Citation Verification (Blind Two-Step)

Verify citations in a research document using the blind verification pattern:
reader agents fetch papers without seeing claims, checker agents compare summaries against claims without seeing sources.

## Constants

- TEMPLATE_DIR = `~/Mao/claude-toolkit/templates/verify-citations`
- READER_MODEL = `haiku`
- CHECKER_MODEL = `haiku`
- QUOTE_VERIFIER_MODEL = `sonnet` (for direct quotes only)

## Context: $ARGUMENTS

If no file path provided, look for `main.tex` in the current directory.

## Workflow

### Step 1: Extract citations and claims

Read the target document. For each `\cite{key}` (LaTeX) or `[key]` (markdown):
1. Extract the **claim** made about the cited paper (the sentence or phrase containing the citation)
2. Classify the claim type:
   - **QUOTE**: Direct quotation (in quotation marks) — highest priority
   - **NUMBER**: Specific quantitative claim (percentages, counts, etc.)
   - **CHARACTERIZATION**: Description of what the paper does or finds
   - **SEE-ALSO**: General reference, no specific claim — skip verification
3. Read the `.bib` file (or reference list) to get paper metadata and URLs

### Step 2: Construct URLs

For each paper, construct fetch URLs from the bib entry:
- arXiv: `https://arxiv.org/abs/{id}` (primary), `https://arxiv.org/html/{id}v1` (fallback for full text)
- DOI: `https://doi.org/{doi}`
- Semantic Scholar: `https://api.semanticscholar.org/graph/v1/paper/{source}:{id}?fields=title,authors,year,venue,abstract`
- If no URL derivable, use OpenAlex: `https://api.openalex.org/works?search="{title}"&per_page=3`

### Step 3: Launch reader agents (parallel)

For each paper with a QUOTE, NUMBER, or CHARACTERIZATION claim, use Bash to fill the reader template:

```
python3 -c "
t = open('$TEMPLATE_DIR/reader.txt').read()
print(t.format(url='PRIMARY_URL', url_fallback='FALLBACK_URL'))
"
```

Launch a **haiku** Agent with the output. Launch ALL in parallel.

**For QUOTE claims**: also launch a **sonnet** Agent that fetches the paper and searches for the exact quoted text. Include the quote in the prompt and ask for verbatim comparison.

**Rate limit mitigation**: If using Semantic Scholar API, stagger launches or prefer arXiv/DOI URLs. Max ~5 concurrent Semantic Scholar requests.

### Step 4: Launch checker agents (parallel)

After ALL readers return, for each paper use Bash to fill the checker template:

```
python3 -c "
t = open('$TEMPLATE_DIR/checker.txt').read()
print(t.format(blind_summary='''PASTE_BLIND_SUMMARY''', claim='''PASTE_CLAIM'''))
"
```

Launch a **haiku** Agent. Launch ALL in parallel.

For QUOTE claims, the checker prompt must include:
- The exact quoted text from the document
- The actual text found by the reader/quote-verifier
- Whether the quote is verbatim, paraphrased, or fabricated

### Step 5: Also check bib entries

While checkers run, verify bib metadata against reader summaries:
- Author names match (first name, last name, completeness)
- Year matches
- Venue matches
- arXiv ID resolves to the correct paper (flag if it resolves to a different paper)

### Step 6: Compile report

**Format**:

```
## Citation Verification Report for `{filename}`

### FLAGGED — Quote Issues
| # | Citation | Issue | Correction |

### FLAGGED — Bib Entry Errors
| # | Citation | Issue | Fix |

### FLAGGED — Claim Accuracy
| # | Citation | Issue |

### VERIFIED (X/Y, no issues)

### COULDN'T VERIFY (list with reasons)
```

For each flagged item, also output in machine-patchable format:
```
FLAGGED | old_text: "exact text in document" | correction: "what it should say" | reason: why
```

### Step 7: Fix (if user requests)

Apply all flagged corrections to the document and bib file. For items that couldn't be verified, add TODO comments.

## Key Rules

- NEVER use Codex MCP for fetching papers — use WebFetch only
- Reader agents must NOT see the claim (blind verification pattern)
- Checker agents must NOT see the source URL (pure text comparison)
- Prioritize QUOTE and NUMBER claims over CHARACTERIZATION
- For wrong arXiv IDs, search for the correct paper via OpenAlex/Google Scholar before flagging
- Flag "can't verify" separately from "verified wrong" — these are different
- Include bib entry checks (authors, year, venue, arXiv ID) alongside claim checks
