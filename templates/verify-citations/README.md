# Citation Verification Pipeline

Two complementary approaches for verifying bibliography entries.

## 1. API-based verification (`verify_bib_api.py`)

**For: verifying metadata (title, authors, year, venue) of BibTeX entries.**

Cascading strategy with zero LLM involvement:
1. **arXiv API** (by ID) — deterministic, for preprints
2. **CrossRef API** (by DOI) — deterministic, for published papers
3. **CrossRef bibliographic search** — excellent for published papers without DOIs
4. **OpenAlex search** — broadest coverage (250M+ works), universal fallback
5. **PDF fallback** — reads page 1 of actual paper PDFs for entries not found by any API

```bash
# Basic usage
python3 verify_bib_api.py references.bib

# With PDF fallback for unpublished papers
python3 verify_bib_api.py references.bib --pdf-dir /path/to/papers/

# Custom output
python3 verify_bib_api.py references.bib --output verification_report.md
```

**Coverage**: Typically 80-90% of entries verified via APIs, remaining via PDFs.

### Why not use LLMs for verification?

Learned the hard way (Validation Dilemma paper, April 2026):
- Both Sonnet and Gemini hallucinate metadata: wrong titles, wrong author names
- They confuse papers by the same author (Trott 2024: two different papers, both LLMs picked the wrong one)
- They "correct" entries to a different wrong answer with high confidence
- APIs queried by unique ID (arXiv, DOI) are deterministic — zero hallucination risk
- For papers without IDs, reading page 1 of the actual PDF is the only reliable method

### To maximize API coverage

Add arXiv IDs and DOIs to every bib entry before running. Most failures come from entries lacking identifiers, forcing fuzzy title search which can match the wrong paper.

## 2. LLM-based claim verification (agent templates)

**For: verifying that claims ABOUT papers are correct (not metadata, but content).**

Uses blind reader/checker pattern to verify claims like "Park et al. report r = 0.66":
- `coordinator.txt` — orchestrates the verification
- `reader.txt` — reads the paper URL without seeing the claim (prevents confirmation bias)
- `checker.txt` — compares blind summary against claim without seeing the source
- `formatter.txt` / `summarizer.txt` — formats results

This is the right use of LLMs in verification: they read and summarize content. They should NOT be used to verify metadata (titles, authors, years).

## Decision guide

| Task | Use |
|------|-----|
| Verify title/authors/year/venue | `verify_bib_api.py` |
| Verify claims about paper content | Agent templates (reader → checker) |
| Find missing papers | Web search + PDF reading |
| Generate bib entries from scratch | API lookup, NEVER LLM generation |
