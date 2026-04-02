# Citation Verification

For each citation in the paper, verify that the attributed claim matches the
cited work's actual content.

## Files to read

[MAIN_TEX, SI_TEX, REFERENCES_BIB]

## For each \cite{} / \citet{} / \citep{}:

1. Find the claim being attributed
2. Read the bib entry (title, venue, year)
3. For methodological claims ("X used Y approach", "X found Z result"),
   web search the paper title to verify

Focus on METHODOLOGICAL claims — not just whether the citation exists.

## Output format

```
## Verified ([count])
- \cite{key}: "claim" — plausible based on title/venue

## Suspicious ([count])
- \cite{key}: "claim" — title suggests [different topic], possible misattribution
  [detail on what the paper actually seems to be about]

## Cannot verify ([count])
- \cite{key}: "claim" — insufficient information to check
```

Check ALL citations. One line per verified, more detail for suspicious.
