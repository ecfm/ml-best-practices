#!/usr/bin/env python3
"""
API-Based Bibliography Verification — Ground Truth Pipeline

Verifies BibTeX entries against authoritative APIs. No LLMs involved.

Strategy (cascading, stops on first match):
  1. arXiv API (by ID) — for preprints
  2. CrossRef API (by DOI) — for published papers with DOIs
  3. CrossRef bibliographic search — for published papers without DOIs
  4. OpenAlex search — universal fallback

For entries not found by any API, falls back to PDF verification
if a papers directory is provided.

Usage:
    python3 verify_bib_api.py references.bib
    python3 verify_bib_api.py references.bib --pdf-dir /path/to/papers/
    python3 verify_bib_api.py references.bib --output report.md
    python3 verify_bib_api.py references.bib --fix  # auto-fix trivial issues

Requires: pdftotext (from poppler-utils) for PDF fallback

Lessons learned (Validation Dilemma paper, 2026-04):
  - LLMs (Sonnet, Gemini) hallucinate metadata: wrong titles, wrong authors,
    wrong papers by the same author. NEVER use LLMs for bib verification.
  - APIs queried by unique ID (arXiv ID, DOI) are deterministic and reliable.
  - CrossRef bibliographic search is excellent for published papers.
  - OpenAlex has the broadest coverage (250M+ works).
  - For unpublished working papers not in any API, read page 1 of the actual PDF.
  - The Trott 2024 case: both Gemini and Sonnet "corrected" a wrong title to a
    DIFFERENT wrong title (different paper by same author). Only the PDF caught it.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path


# ── LaTeX cleaning ──────────────────────────────────────────────────

def clean_latex(text: str) -> str:
    """Clean LaTeX markup, preserving text inside braces."""
    text = re.sub(r"\\['\"`^~][{]?(\w)[}]?", r'\1', text)  # accents
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── BibTeX parsing ──────────────────────────────────────────────────

def parse_bib(text: str) -> list[dict]:
    """Parse BibTeX into structured entries."""
    entries = []
    blocks = re.split(r'\n(?=@)', '\n' + text)

    for block in blocks:
        block = block.strip()
        if not block.startswith('@'):
            continue
        m = re.match(r'@(\w+)\{(\w+),', block)
        if not m:
            continue

        key = m.group(2)

        def get_field(name):
            pattern = rf'{name}\s*=\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}'
            fm = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
            return fm.group(1).strip() if fm else ""

        title_raw = get_field("title")
        author_raw = get_field("author")
        journal = get_field("journal")
        booktitle = get_field("booktitle")
        doi = get_field("doi")
        note = get_field("note")

        title = clean_latex(title_raw)

        # Extract arXiv ID from anywhere
        arxiv_id = None
        for field in [journal, note, block]:
            am = re.search(r'arXiv[: ]*(\d{4}\.\d{4,5})', field)
            if am:
                arxiv_id = am.group(1)
                break

        # Parse first author last name
        first_author = ""
        if author_raw:
            first_part = clean_latex(author_raw.split(" and ")[0].strip())
            first_author = first_part.split(",")[0].strip() if "," in first_part else first_part.split()[-1]

        # All authors
        authors = [clean_latex(a.strip()) for a in author_raw.split(" and ") if a.strip().lower() != "others"]

        entries.append({
            "key": key,
            "title": title,
            "authors": authors,
            "first_author": first_author,
            "year": get_field("year"),
            "venue": journal or booktitle,
            "doi": clean_latex(doi),
            "arxiv_id": arxiv_id,
            "volume": get_field("volume"),
            "pages": get_field("pages"),
        })
    return entries


# ── API queries ─────────────────────────────────────────────────────

def api_get(url: str) -> dict | str | None:
    """HTTP GET with retry on 429."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "BibVerify/2.0 (mailto:verify@example.com)")
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(3)
                continue
            return None
        except Exception:
            return None
    return None


def query_arxiv(arxiv_id: str) -> dict | None:
    """Query arXiv API by ID. Deterministic and reliable."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    xml_text = api_get(url)
    if not xml_text or not isinstance(xml_text, str):
        return None
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = re.sub(r'\s+', ' ', entry.findtext("atom:title", "", ns).strip())
        authors = [a.findtext("atom:name", "", ns).strip()
                   for a in entry.findall("atom:author", ns)]
        year = entry.findtext("atom:published", "", ns)[:4]
        return {"title": title, "authors": authors, "year": year, "source": "arxiv"}
    except ET.ParseError:
        return None


def query_crossref_doi(doi: str) -> dict | None:
    """Query CrossRef by DOI. Deterministic and authoritative."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    data = api_get(url)
    if not data or not isinstance(data, dict):
        return None
    msg = data.get("message", {})
    title = (msg.get("title") or [""])[0]
    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
               for a in msg.get("author", [])]
    year = ""
    dp = msg.get("published-print", msg.get("published-online", {}))
    if dp and dp.get("date-parts"):
        year = str(dp["date-parts"][0][0])
    return {"title": title, "authors": authors, "year": year, "source": "crossref-doi"}


def query_crossref_search(title: str, first_author: str, year: str) -> dict | None:
    """Search CrossRef by bibliographic query. Excellent for published papers."""
    q = f"{first_author} {title[:100]}"
    params = urllib.parse.urlencode({"query.bibliographic": q, "rows": 3})
    data = api_get(f"https://api.crossref.org/works?{params}")
    if not data or not isinstance(data, dict):
        return None
    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    best, best_ratio = None, 0
    for item in items:
        item_title = (item.get("title") or [""])[0].lower()
        ratio = SequenceMatcher(None, title.lower(), item_title).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = item

    if best and best_ratio > 0.7:
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                   for a in best.get("author", [])]
        year_found = ""
        dp = best.get("published-print", best.get("published-online", {}))
        if dp and dp.get("date-parts"):
            year_found = str(dp["date-parts"][0][0])
        return {
            "title": (best.get("title") or [""])[0],
            "authors": authors, "year": year_found,
            "source": f"crossref-search (sim={best_ratio:.2f})",
        }
    return None


def query_openalex(title: str) -> dict | None:
    """Search OpenAlex. Broadest coverage (250M+ works)."""
    q = urllib.parse.quote(title[:150])
    data = api_get(f"https://api.openalex.org/works?search={q}&per_page=3")
    if not data or not isinstance(data, dict):
        return None

    best, best_ratio = None, 0
    for r in data.get("results", []):
        r_title = (r.get("title") or "").lower()
        ratio = SequenceMatcher(None, title.lower(), r_title).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = r

    if best and best_ratio > 0.7:
        authors = [a["author"]["display_name"]
                   for a in best.get("authorships", [])
                   if a.get("author", {}).get("display_name")]
        year = str(best.get("publication_year", ""))
        return {"title": best.get("title", ""), "authors": authors,
                "year": year, "source": f"openalex (sim={best_ratio:.2f})"}
    return None


# ── PDF fallback ────────────────────────────────────────────────────

def verify_against_pdf(entry: dict, pdf_dir: Path) -> dict | None:
    """Try to find and verify against actual PDF."""
    pdfs = list(pdf_dir.glob("*.pdf"))
    first_author_lower = entry["first_author"].lower()

    # Find PDF by first author name in filename
    candidates = [p for p in pdfs if first_author_lower in p.stem.lower().replace("_", " ")]
    if not candidates:
        return None

    # Use title words to disambiguate
    title_words = set(re.findall(r'\w{4,}', entry["title"].lower()))
    best, best_score = None, 0
    for pdf in candidates:
        fname_words = set(re.findall(r'\w{4,}', pdf.stem.lower().replace("_", " ")))
        score = len(title_words & fname_words)
        if score > best_score:
            best_score = score
            best = pdf

    if not best:
        best = candidates[0]

    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", "-layout", str(best), "-"],
            capture_output=True, text=True, timeout=10)
        pdf_text = result.stdout.lower()
    except Exception:
        return None

    # Check title words in PDF
    title_words = [w for w in re.findall(r'\w{4,}', entry["title"].lower()) if w not in {"with", "from", "that", "this", "have", "been"}]
    if not title_words:
        return None

    matches = sum(1 for w in title_words if w in pdf_text)
    ratio = matches / len(title_words) if title_words else 0

    if ratio >= 0.6:
        return {"source": f"pdf ({best.name})", "title_match_ratio": ratio}
    else:
        # Get first substantive line from PDF for error reporting
        lines = [l.strip() for l in result.stdout.split("\n") if l.strip() and len(l.strip()) > 15]
        pdf_title = lines[0] if lines else "?"
        return {"source": f"pdf ({best.name})", "title_match_ratio": ratio,
                "pdf_title": pdf_title, "mismatch": True}


# ── Comparison ──────────────────────────────────────────────────────

def compare(entry: dict, api: dict) -> list[str]:
    """Compare bib entry against API result. Return list of issues."""
    issues = []

    # Title
    if api.get("title"):
        bt = re.sub(r'[^a-z0-9 ]', '', entry["title"].lower())
        at = re.sub(r'[^a-z0-9 ]', '', api["title"].lower())
        ratio = SequenceMatcher(None, bt, at).ratio()
        if ratio < 0.80:
            issues.append(f"Title (sim={ratio:.2f}): bib='{entry['title'][:60]}' vs api='{api['title'][:60]}'")

    # Authors (check last names)
    if api.get("authors"):
        for bib_a in entry["authors"][:10]:
            bib_last = bib_a.split(",")[0].strip().lower() if "," in bib_a else bib_a.split()[-1].lower()
            found = any(bib_last in api_a.lower() for api_a in api["authors"])
            if not found:
                issues.append(f"Author '{bib_a}' not found in API list")

    # Year (allow ±1)
    if api.get("year") and entry["year"]:
        try:
            if abs(int(api["year"]) - int(entry["year"])) > 1:
                issues.append(f"Year: bib={entry['year']}, api={api['year']}")
        except ValueError:
            pass

    return issues


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify BibTeX against APIs + PDFs")
    parser.add_argument("bib_file", type=Path)
    parser.add_argument("--pdf-dir", type=Path, default=None,
                        help="Directory with paper PDFs for fallback verification")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output report path (default: bib_verification.md)")
    args = parser.parse_args()

    if not args.output:
        args.output = args.bib_file.parent / "bib_verification.md"

    entries = parse_bib(args.bib_file.read_text())
    print(f"Parsed {len(entries)} entries "
          f"({sum(1 for e in entries if e['arxiv_id'])} arXiv, "
          f"{sum(1 for e in entries if e['doi'])} DOI)")

    ok, issues, not_found, pdf_verified = [], [], [], []

    for i, entry in enumerate(entries):
        api_data = None

        # Cascade: arXiv → CrossRef DOI → CrossRef search → OpenAlex
        if entry["arxiv_id"]:
            api_data = query_arxiv(entry["arxiv_id"])
            time.sleep(0.4)
        if not api_data and entry["doi"]:
            api_data = query_crossref_doi(entry["doi"])
            time.sleep(0.3)
        if not api_data and entry["venue"] and entry["first_author"]:
            api_data = query_crossref_search(entry["title"], entry["first_author"], entry["year"])
            time.sleep(0.3)
        if not api_data:
            api_data = query_openalex(entry["title"])
            time.sleep(0.3)

        if api_data:
            entry_issues = compare(entry, api_data)
            src = api_data.get("source", "?")
            if entry_issues:
                issues.append({"key": entry["key"], "issues": entry_issues, "source": src})
                print(f"  [{i+1}/{len(entries)}] {entry['key']}: ISSUE ({src})")
            else:
                ok.append({"key": entry["key"], "source": src})
                print(f"  [{i+1}/{len(entries)}] {entry['key']}: OK ({src})")
        else:
            # PDF fallback
            if args.pdf_dir:
                pdf_result = verify_against_pdf(entry, args.pdf_dir)
                if pdf_result and not pdf_result.get("mismatch"):
                    pdf_verified.append({"key": entry["key"], "source": pdf_result["source"]})
                    print(f"  [{i+1}/{len(entries)}] {entry['key']}: OK ({pdf_result['source']})")
                elif pdf_result and pdf_result.get("mismatch"):
                    issues.append({
                        "key": entry["key"],
                        "issues": [f"PDF title mismatch: '{pdf_result.get('pdf_title', '?')[:60]}'"],
                        "source": pdf_result["source"],
                    })
                    print(f"  [{i+1}/{len(entries)}] {entry['key']}: ISSUE ({pdf_result['source']})")
                else:
                    not_found.append(entry["key"])
                    print(f"  [{i+1}/{len(entries)}] {entry['key']}: NOT FOUND")
            else:
                not_found.append(entry["key"])
                print(f"  [{i+1}/{len(entries)}] {entry['key']}: NOT FOUND")

    # Report
    lines = [
        "# Bibliography Verification Report\n",
        f"Verified {len(entries)} entries via arXiv → CrossRef → OpenAlex → PDF fallback.\n",
        f"**OK: {len(ok)} | PDF OK: {len(pdf_verified)} | Issues: {len(issues)} | Not found: {len(not_found)}**\n",
    ]

    if issues:
        lines.append(f"\n## Issues ({len(issues)})\n")
        for r in issues:
            lines.append(f"### {r['key']} ({r['source']})")
            for iss in r["issues"]:
                lines.append(f"- {iss}")
            lines.append("")

    if not_found:
        lines.append(f"\n## Not Found ({len(not_found)})\n")
        for k in not_found:
            lines.append(f"- {k}")

    lines.append(f"\n## Verified OK — API ({len(ok)})\n")
    for r in ok:
        lines.append(f"- {r['key']} ({r['source']})")

    if pdf_verified:
        lines.append(f"\n## Verified OK — PDF ({len(pdf_verified)})\n")
        for r in pdf_verified:
            lines.append(f"- {r['key']} ({r['source']})")

    args.output.write_text("\n".join(lines))
    total_ok = len(ok) + len(pdf_verified)
    print(f"\nAPI OK: {len(ok)}, PDF OK: {len(pdf_verified)}, Issues: {len(issues)}, Not found: {len(not_found)}")
    print(f"Coverage: {total_ok}/{len(entries)} ({100*total_ok/len(entries):.0f}%)")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
