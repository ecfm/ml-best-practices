---
name: deep-lit-review
description: Orchestrated multi-agent deep literature review with iterative follow-up, citation graph traversal, and convergence-based stopping. Use when user says "deep lit review", "review the literature", "find all papers on X", "lit review agents", or needs comprehensive literature exploration beyond simple search.
argument-hint: [topic-or-list-of-questions] — output: [path] — max-rounds: [N]
allowed-tools: Agent, Read, Write, Glob, Grep, Bash(*), WebSearch, WebFetch
---

# Deep Literature Review — Orchestrated Multi-Agent

## Input Parsing

Parse `$ARGUMENTS` for:
- **Topic or questions**: The research area or specific list of questions to investigate
- `— output: [path]`: Where to save results (default: `docs/lit_review/{date}-deep-review/`)
- `— max-rounds: [N]`: Maximum follow-up iterations (default: 10)
- `— agents: [N]`: Number of initial parallel agents (default: 7)
- `— model: [sonnet|haiku|opus]`: Model for search agents (default: sonnet)

## File Structure

The skill creates two separate areas — agent-facing working files and human-facing reports:

```
{output_dir}/
├── workbench/                          # Agent-facing (raw, machine-readable)
│   ├── agents/                         # Each initial agent's working output
│   │   ├── {id}_{short_name}.md
│   │   └── ...
│   ├── follow_ups/                     # Follow-up agent outputs
│   │   ├── f{round}_{id}_{topic}.md
│   │   └── ...
│   └── orchestrator_log.md             # All decisions: what launched, why, what came back
│
├── reports/                            # Human-facing (clean, distilled)
│   ├── per_stream/                     # One page per research stream
│   │   ├── {stream_name}.md
│   │   └── ...
│   ├── synthesis.md                    # Cross-stream synthesis + direction recommendations
│   └── open_questions.md               # What we still don't know
│
├── state.md                            # Live status dashboard
└── README.md                           # Overview for the human
```

## Orchestration Loop

```
round = 0

WHILE round < max_rounds AND NOT converged:

    IF round == 0:
        Launch initial batch of agents (parallel, sonnet)
        Each agent follows the AGENT SEARCH PROTOCOL (see templates/agent_prompt.md)
    ELSE:
        Launch follow-up agents based on review decisions

    WAIT for agents to complete

    REVIEW each agent result:
        Read TL;DR section only (Level 1 review)
        For each stream, decide:
            DEEPEN  — promising lead not fully followed
                    → follow-up agent on specific sub-question
            EXPAND  — discovered sub-field none of the streams cover
                    → new agent with new question
            MERGE   — two streams found overlapping territory
                    → one follow-up synthesizing both
            RESOLVE — two streams contradict
                    → targeted agent to find deciding evidence
            CLOSE   — question answered with sufficient confidence
                    → write report for this stream

    UPDATE state.md with decisions

    CHECK convergence (see CONVERGENCE CRITERIA below)

    round += 1

AFTER loop exits:
    Launch SYNTHESIS AGENT (reads all workbench files, writes reports/)
    Update README.md with final status
```

## Convergence Criteria

**Quantitative (measured via OpenAlex + search results):**
- **New paper rate**: Track unique papers found per round. If last 2 rounds found <3 new relevant papers combined → converging
- **Citation frontier exhausted**: For each key paper, both its references AND citers have been checked. No unexplored high-citation nodes remain
- **Cross-stream stability**: Stream TL;DRs haven't changed substantively in 2 rounds

**Qualitative (judged by orchestrator):**
- All initial questions have either a confident answer or a documented "insufficient evidence"
- No unresolved contradictions between streams
- No high-priority expansion leads left unfollowed

**Early stop triggers (can stop before max rounds):**
- All streams reach CLOSED status
- 3 consecutive rounds with <2 new relevant papers total
- All expansion leads have been followed or deprioritized

**Late stop protection (keep going even if seemingly converged):**
- A key paper was found in last round that opens new direction → don't stop
- Two streams contradict and resolution agent hasn't run → don't stop
- A stream returned "nothing exists" but only tried 1-2 query formulations → verify before stopping

## Citation Graph Traversal

Agents use OpenAlex API for programmatic citation traversal. Rate limit: 10,000 requests/window (~20 hours), $0.001/request.

**API patterns (baked into agent prompts):**
```bash
# Search papers by topic
curl -s "https://api.openalex.org/works?search=QUERY&per_page=10&select=id,title,publication_year,cited_by_count,doi"

# Get a paper's references (backward traversal)
curl -s "https://api.openalex.org/works/OPENALEX_ID?select=title,referenced_works"

# Get papers citing a paper (forward traversal)
curl -s "https://api.openalex.org/works?filter=cites:OPENALEX_ID&per_page=10&sort=cited_by_count:desc&select=id,title,publication_year,cited_by_count"

# Batch resolve OpenAlex IDs to metadata
curl -s "https://api.openalex.org/works?filter=openalex:ID1|ID2|ID3&select=id,title,publication_year,cited_by_count,doi"
```

**Traversal strategy:**
- Forward citations (who cites this?) → finds recent follow-up work
- Backward citations (what does this cite?) → finds foundational work
- OpenAlex for graph structure; WebFetch for reading related work sections (authors' contextual commentary)
- WebSearch for very recent preprints not yet indexed by OpenAlex

## Model Allocation

| Task | Model | Rationale |
|------|-------|-----------|
| Search agents (all rounds) | sonnet | Needs judgment for relevance, not just extraction |
| Follow-up agents | sonnet | Same — evaluation matters |
| Synthesis agent | sonnet | Cross-references all streams; runs in own context |
| Final review in main conversation | opus (parent) | User interaction, strategic decisions |

## Tiered Reading (Context Management)

The orchestrator in the main conversation never reads full agent outputs. Instead:

```
Level 0: state.md only (~50 lines)
    → Enough to decide what to launch next

Level 1: TL;DR section of each agent file (~10 lines each)
    → Enough to review quality and decide DEEPEN/EXPAND/CLOSE

Level 2: Full agent file (~200+ lines)
    → Only for streams needing DEEPEN or RESOLVE

Level 3: Fetch actual paper
    → Only for papers pivotal to a decision
```

The synthesis agent gets its own fresh context and can read all Level 2 files.
