#!/usr/bin/env python3
"""Round manager: orchestrates one fix→review→fix→review iteration.

Each round:
  1. Load prior rounds → build accumulated known issues
  2. Run Phase 0 (check_numbers + check_staleness)
  3. Generate prompts with known issues injected
  4. Run LLM reviews
  5. Run consolidation
  6. Create round record (YAML) for triage
  7. After fixes: run Phase 0 re-check (--verify)

Usage:
    python run_round.py                    # start new round
    python run_round.py --verify           # post-fix verification only
    python run_round.py --status           # show iteration history
    python run_round.py --resolve r3_001 fixed "Updated main.tex line 88"

State lives in rounds/ directory. Each round is a YAML file.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml required: pip install pyyaml")
    sys.exit(1)

REVIEW_DIR = Path(__file__).resolve().parent
ROUNDS_DIR = REVIEW_DIR / "rounds"
PHASE0_DIR = REVIEW_DIR / "phase0"
CONFIG_PATH = REVIEW_DIR / "project_config.yaml"


def get_round_number():
    """Next round number based on existing files."""
    existing = list(ROUNDS_DIR.glob("round_*.yaml"))
    if not existing:
        return 1
    nums = [int(f.stem.split("_")[1]) for f in existing]
    return max(nums) + 1


def load_all_rounds():
    """Load all round records in order."""
    rounds = []
    for f in sorted(ROUNDS_DIR.glob("round_*.yaml")):
        with open(f) as fh:
            rounds.append(yaml.safe_load(fh))
    return rounds


def build_known_issues(rounds):
    """Accumulate known issues from all prior rounds."""
    known = []
    for r in rounds:
        for f in r.get("findings", []):
            status = f.get("status", "open")
            if status in ("fixed", "disclosed", "rejected"):
                desc = f["description"]
                if status == "fixed":
                    known.append(f"{desc} (fixed round {r['round']})")
                elif status == "disclosed":
                    known.append(f"{desc} (disclosed)")
    return known


def run_phase0():
    """Run Phase 0 checks, return results."""
    results = {"numbers": None, "staleness": None}

    num_script = PHASE0_DIR / "check_numbers.py"
    stale_script = PHASE0_DIR / "check_staleness.py"

    if num_script.exists():
        r = subprocess.run(
            [sys.executable, str(num_script), "--brief"],
            capture_output=True, text=True
        )
        results["numbers"] = {
            "exit_code": r.returncode,
            "output": r.stdout.strip(),
        }
        print(r.stdout)

    if stale_script.exists():
        r = subprocess.run(
            [sys.executable, str(stale_script)],
            capture_output=True, text=True
        )
        results["staleness"] = {
            "exit_code": r.returncode,
            "output": r.stdout.strip(),
        }
        print(r.stdout)

    return results


def create_round_record(round_num, phase0_results, review_dir):
    """Create a round YAML for triage."""
    record = {
        "round": round_num,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "open",  # open → in_progress → verified → closed
        "phase0": {
            "numbers_clean": phase0_results.get("numbers", {}).get("exit_code") == 0,
            "staleness_clean": phase0_results.get("staleness", {}).get("exit_code") == 0,
        },
        "reviews": [],
        "findings": [],
        "convergence": {
            "must_fix": 0,
            "should_fix": 0,
            "disclosed": 0,
            "note": "Fill in after triage",
        },
    }

    # List review files
    if review_dir.exists():
        for f in sorted(review_dir.glob("*.md")):
            record["reviews"].append(f.name)

    return record


def show_status(rounds):
    """Print iteration history."""
    print("=== Round History ===\n")
    for r in rounds:
        status = r.get("status", "unknown")
        findings = r.get("findings", [])
        n_must = sum(1 for f in findings if f.get("severity") == "must_fix")
        n_should = sum(1 for f in findings if f.get("severity") == "should_fix")
        n_fixed = sum(1 for f in findings if f.get("status") == "fixed")
        n_open = sum(1 for f in findings if f.get("status") == "open")

        print(f"Round {r['round']} ({r.get('date', '?')}): {status}")
        print(f"  Findings: {len(findings)} total, {n_must} must-fix, {n_should} should-fix")
        print(f"  Resolved: {n_fixed} fixed, {n_open} open")
        p0 = r.get("phase0", {})
        print(f"  Phase 0: numbers={'clean' if p0.get('numbers_clean') else 'issues'}, "
              f"staleness={'clean' if p0.get('staleness_clean') else 'issues'}")
        print()

    # Convergence trend
    if len(rounds) >= 2:
        must_fixes = [
            sum(1 for f in r.get("findings", []) if f.get("severity") == "must_fix")
            for r in rounds
        ]
        print(f"Convergence trend (must-fix): {' → '.join(str(n) for n in must_fixes)}")
        if must_fixes[-1] == 0:
            print("  ✓ Converged: 0 must-fix items in latest round")
        elif len(must_fixes) >= 2 and must_fixes[-1] < must_fixes[-2]:
            print(f"  Improving: {must_fixes[-2]} → {must_fixes[-1]}")
        else:
            print(f"  Not converging: consider deeper investigation")


def resolve_finding(round_file, finding_id, status, resolution):
    """Update a finding's status in a round record."""
    with open(round_file) as f:
        record = yaml.safe_load(f)

    found = False
    for finding in record.get("findings", []):
        if finding.get("id") == finding_id:
            finding["status"] = status
            finding["resolution"] = resolution
            finding["resolved_date"] = datetime.now().strftime("%Y-%m-%d")
            found = True
            break

    if not found:
        print(f"Finding {finding_id} not found in {round_file.name}")
        sys.exit(1)

    with open(round_file, "w") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {finding_id} → {status}")


def main():
    parser = argparse.ArgumentParser(description="Paper review round manager")
    parser.add_argument("--verify", action="store_true",
                        help="Post-fix verification only (Phase 0 re-check)")
    parser.add_argument("--status", action="store_true",
                        help="Show iteration history and convergence")
    parser.add_argument("--resolve", nargs=3, metavar=("FINDING_ID", "STATUS", "RESOLUTION"),
                        help="Resolve a finding: --resolve r2_001 fixed 'Updated line 88'")
    parser.add_argument("--phase0-only", action="store_true",
                        help="Run Phase 0 only, no LLM reviews")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    rounds = load_all_rounds()

    # ── Status ──
    if args.status:
        if not rounds:
            print("No rounds yet. Run: python run_round.py")
        else:
            show_status(rounds)
        return

    # ── Resolve a finding ──
    if args.resolve:
        finding_id, status, resolution = args.resolve
        # Find which round contains this finding
        for r in reversed(rounds):
            round_file = ROUNDS_DIR / f"round_{r['round']}.yaml"
            for f in r.get("findings", []):
                if f.get("id") == finding_id:
                    resolve_finding(round_file, finding_id, status, resolution)
                    return
        print(f"Finding {finding_id} not found in any round")
        sys.exit(1)

    # ── Verify (post-fix) ──
    if args.verify:
        print("=== Post-Fix Verification ===\n")
        results = run_phase0()
        nums_ok = results.get("numbers", {}).get("exit_code") == 0
        stale_ok = results.get("staleness", {}).get("exit_code") == 0

        if nums_ok and stale_ok:
            print("\n✓ All Phase 0 checks pass. Ready for next round or submission.")
            # Update latest round status
            if rounds:
                latest_file = ROUNDS_DIR / f"round_{rounds[-1]['round']}.yaml"
                with open(latest_file) as f:
                    record = yaml.safe_load(f)
                record["status"] = "verified"
                record["verified_date"] = datetime.now().strftime("%Y-%m-%d")
                with open(latest_file, "w") as f:
                    yaml.dump(record, f, default_flow_style=False, sort_keys=False)
        else:
            print("\n✗ Phase 0 issues remain. Fix before proceeding.")
            sys.exit(1)
        return

    # ── New round ──
    round_num = get_round_number()
    print(f"=== Starting Round {round_num} ===\n")

    # Step 1: Phase 0
    print("--- Phase 0: Automated Checks ---\n")
    phase0_results = run_phase0()

    if args.phase0_only:
        return

    # Step 2: Generate prompts with known issues
    known = build_known_issues(rounds)
    if known:
        print(f"\n--- Injecting {len(known)} known issues from prior rounds ---\n")

    # Step 3: Generate prompts
    if CONFIG_PATH.exists():
        fill_cmd = [
            sys.executable, str(REVIEW_DIR / "fill_templates.py"),
            str(CONFIG_PATH),
            "--out-dir", str(REVIEW_DIR / "prompts_filled"),
        ]
        if known:
            fill_cmd.append("--inject-known")
        subprocess.run(fill_cmd)

    # Step 4: Run reviews
    review_out = REVIEW_DIR / "reviews" / f"round_{round_num}"
    review_out.mkdir(parents=True, exist_ok=True)

    launch_script = REVIEW_DIR / "launch.py"
    if launch_script.exists():
        print(f"\n--- Running LLM Reviews (parallel={args.parallel}) ---\n")
        subprocess.run([
            sys.executable, str(launch_script),
            "--parallel", str(args.parallel),
            "--timeout", str(args.timeout),
        ])

    # Step 5: Create round record
    record = create_round_record(round_num, phase0_results, review_out)
    round_file = ROUNDS_DIR / f"round_{round_num}.yaml"
    with open(round_file, "w") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)

    print(f"\n=== Round {round_num} Complete ===")
    print(f"Round record: {round_file}")
    print(f"Reviews: {review_out}/")
    print(f"\nNext steps:")
    print(f"  1. Read reviews and add findings to {round_file.name}")
    print(f"  2. For each finding: python run_round.py --resolve r{round_num}_001 fixed 'description'")
    print(f"  3. After all fixes: python run_round.py --verify")
    print(f"  4. Check convergence: python run_round.py --status")


if __name__ == "__main__":
    main()
