# Execute & Verify: [PIPELINE_NAME]

Run `[SCRIPT_PATH]` end-to-end. Compare its outputs against the manuscript
values listed below.

## Setup

```bash
cd [REPO_DIR]
[RUN_COMMAND]
```

If it fails, report the failure before attempting fixes.

## Expected values

| Metric | Expected | Source |
|--------|----------|--------|
| [METRIC_1] | [VALUE_1] | [TEX_LOCATION] |
| [METRIC_2] | [VALUE_2] | [TEX_LOCATION] |

Also check: [OUTPUT_FILES]

## Deliverable

Report ONLY:
1. **Execution failures** — did the script run? Any errors?
2. **Number mismatches** — script output vs expected values above
3. **Untracked outputs** — files the script produces that aren't in provenance

Do NOT report code style, documentation, or methodology opinions.

```
## Execution Result
[pass/fail + errors]

## Mismatches
| Metric | Expected | Got | File |

## Untracked Outputs
[list]
```
