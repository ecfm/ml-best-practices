#!/usr/bin/env python3
"""Generate evidence graph from claims files.

Template from claude-toolkit/templates/claims-tracker/
Customize the CONFIG dict below for your project.

Reads all claims/*.yaml, extracts supports/depends_on/replicated_by edges,
and outputs:
  - claims/evidence_graph.yaml  (adjacency list for programmatic use)
  - claims/evidence_graph.dot   (DOT format for visualization)

Also identifies: root claims (nothing depends on them), leaf claims (they
support nothing), and the critical path (longest dependency chain).

Usage:
    python scripts/generate_evidence_graph.py
"""

import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: uv pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ── Project configuration ──

CONFIG = {
    "claims_dir": "claims",
}

PAPER = Path(__file__).resolve().parent.parent
CLAIMS_DIR = PAPER / CONFIG["claims_dir"]


def load_all_claims():
    """Load all claims from subdirectories of claims/."""
    all_claims = {}
    if not CLAIMS_DIR.exists():
        return all_claims
    for subdir in sorted(d.name for d in CLAIMS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")):
        d = CLAIMS_DIR / subdir
        for path in sorted(d.glob("*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            for c in data.get("claims", []):
                cid = c.get("id")
                if cid:
                    all_claims[cid] = {
                        "text": c.get("text", "").strip(),
                        "type": c.get("type", "unknown"),
                        "strength": c.get("strength"),
                        "supports": c.get("supports", []),
                        "depends_on": c.get("depends_on", []),
                        "replicated_by": c.get("replicated_by", []),
                        "source_file": path.name,
                    }
    return all_claims


def build_graph(all_claims):
    """Build adjacency lists from claim relationships."""
    # Forward edges: A supports B means A -> B (A provides evidence for B)
    supports_edges = defaultdict(list)    # child -> [parents it supports]
    depends_edges = defaultdict(list)     # parent -> [children it depends on]
    replicated_edges = defaultdict(list)  # claim -> [claims that replicate it]

    for cid, c in all_claims.items():
        for target in c["supports"]:
            supports_edges[cid].append(target)
        for dep in c["depends_on"]:
            depends_edges[cid].append(dep)
        for rep in c["replicated_by"]:
            replicated_edges[cid].append(rep)

    return supports_edges, depends_edges, replicated_edges


def find_roots_and_leaves(all_claims, supports_edges):
    """Find root claims (supported by others, support nothing themselves) and
    leaf claims (support others but nothing supports them)."""
    all_ids = set(all_claims.keys())

    # Claims that appear as targets of supports edges
    supported_by_others = set()
    for cid, targets in supports_edges.items():
        for t in targets:
            supported_by_others.add(t)

    # Claims that support something
    supports_something = set(supports_edges.keys())

    # Roots: supported by others but don't support anything further
    # (top of the argument -- thesis-level claims)
    roots = supported_by_others - supports_something

    # Also add claims that aren't referenced at all in supports
    # (could be the ultimate thesis claims not yet connected)
    unreferenced = all_ids - supported_by_others - supports_something
    if unreferenced:
        roots.update(cid for cid in unreferenced if all_claims[cid]["type"] in ("interpretive", "logical"))

    # Leaves: support something but nothing supports them
    # (bottom of the argument -- raw evidence)
    leaves = supports_something - supported_by_others

    # Also claims with no supports and no one supporting them (isolated)
    isolated = all_ids - supported_by_others - supports_something

    return roots, leaves, isolated


def find_longest_chain(all_claims, supports_edges):
    """Find the longest dependency chain (critical path) using DFS."""
    # Build reverse graph: if A supports B, then B's argument depends on A
    # We want the longest path from leaf to root through supports edges
    reverse = defaultdict(list)
    for cid, targets in supports_edges.items():
        for t in targets:
            reverse[t].append(cid)

    # DFS with memoization
    memo = {}

    def longest_from(node, visited):
        if node in memo:
            return memo[node]
        children = reverse.get(node, [])
        if not children:
            memo[node] = [node]
            return [node]
        best = []
        for child in children:
            if child in visited:
                continue  # avoid cycles
            visited.add(child)
            path = longest_from(child, visited)
            visited.discard(child)
            if len(path) > len(best):
                best = path
        result = [node] + best
        memo[node] = result
        return result

    longest = []
    for cid in all_claims:
        path = longest_from(cid, {cid})
        if len(path) > len(longest):
            longest = path

    return longest


def find_weak_load_bearing(all_claims, supports_edges):
    """Find claims that are weak/contested but support strong claims."""
    issues = []
    for cid, targets in supports_edges.items():
        c_strength = all_claims.get(cid, {}).get("strength")
        if c_strength not in ("weak", "contested"):
            continue
        for target in targets:
            t_strength = all_claims.get(target, {}).get("strength")
            if t_strength in ("strong", "moderate"):
                issues.append((cid, c_strength, target, t_strength))
    return issues


def write_yaml(all_claims, supports_edges, depends_edges, replicated_edges,
               roots, leaves, isolated, longest_chain, weak_lb):
    """Write evidence_graph.yaml."""
    nodes = {}
    for cid, c in all_claims.items():
        node = {
            "text": c["text"][:80],
            "type": c["type"],
        }
        if c.get("strength"):
            node["strength"] = c["strength"]
        if supports_edges.get(cid):
            node["supports"] = supports_edges[cid]
        if depends_edges.get(cid):
            node["depends_on"] = depends_edges[cid]
        if replicated_edges.get(cid):
            node["replicated_by"] = replicated_edges[cid]
        nodes[cid] = node

    graph = {
        "nodes": nodes,
        "analysis": {
            "total_claims": len(all_claims),
            "roots": sorted(roots),
            "leaves": sorted(leaves),
            "isolated": sorted(isolated),
            "longest_chain": longest_chain,
            "longest_chain_length": len(longest_chain),
            "weak_load_bearing": [
                {"claim": wlb[0], "strength": wlb[1],
                 "supports": wlb[2], "target_strength": wlb[3]}
                for wlb in weak_lb
            ],
        },
    }

    out_path = CLAIMS_DIR / "evidence_graph.yaml"
    with open(out_path, "w") as f:
        yaml.dump(graph, f, default_flow_style=False, sort_keys=False, width=120)
    return out_path


def write_dot(all_claims, supports_edges, replicated_edges, roots, leaves, weak_lb):
    """Write evidence_graph.dot for visualization."""
    weak_lb_claims = {wlb[0] for wlb in weak_lb}

    type_colors = {
        "empirical": "#4A90D9",
        "logical": "#7B68EE",
        "literature": "#DAA520",
        "interpretive": "#2ECC71",
        "scope": "#95A5A6",
        "qualification": "#E67E22",
    }
    strength_shapes = {
        "strong": "box",
        "moderate": "ellipse",
        "weak": "diamond",
        "contested": "octagon",
    }

    lines = [
        'digraph evidence {',
        '  rankdir=BT;',
        '  node [fontsize=9, fontname="Helvetica"];',
        '  edge [fontsize=7];',
        '',
    ]

    for cid, c in all_claims.items():
        color = type_colors.get(c["type"], "#CCCCCC")
        shape = strength_shapes.get(c.get("strength"), "ellipse")
        label = cid.replace("_", "\\n")
        style = "bold" if cid in roots else ""
        if cid in weak_lb_claims:
            style = "dashed"
            color = "#E74C3C"
        attr = f'label="{label}", shape={shape}, color="{color}", style="{style}"'
        lines.append(f'  {cid} [{attr}];')

    lines.append('')

    for cid, targets in supports_edges.items():
        for t in targets:
            lines.append(f'  {cid} -> {t} [label="supports"];')

    for cid, reps in replicated_edges.items():
        for r in reps:
            if r in all_claims:
                lines.append(f'  {r} -> {cid} [label="replicates", style=dashed, color=green];')

    lines.append('}')

    out_path = CLAIMS_DIR / "evidence_graph.dot"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    all_claims = load_all_claims()
    if not all_claims:
        print("No claims found.")
        sys.exit(2)

    supports_edges, depends_edges, replicated_edges = build_graph(all_claims)
    roots, leaves, isolated = find_roots_and_leaves(all_claims, supports_edges)
    longest_chain = find_longest_chain(all_claims, supports_edges)
    weak_lb = find_weak_load_bearing(all_claims, supports_edges)

    yaml_path = write_yaml(all_claims, supports_edges, depends_edges,
                           replicated_edges, roots, leaves, isolated,
                           longest_chain, weak_lb)
    dot_path = write_dot(all_claims, supports_edges, replicated_edges,
                         roots, leaves, weak_lb)

    print(f"=== Evidence Graph ===")
    print(f"Claims: {len(all_claims)}")
    print(f"Supports edges: {sum(len(v) for v in supports_edges.values())}")
    print(f"Replication edges: {sum(len(v) for v in replicated_edges.values())}")
    print(f"Roots (thesis-level): {sorted(roots)}")
    print(f"Leaves (raw evidence): {sorted(leaves)}")
    if isolated:
        print(f"Isolated (no edges): {sorted(isolated)}")
    print(f"Longest chain ({len(longest_chain)}): {' -> '.join(longest_chain)}")
    if weak_lb:
        print(f"\nWeak load-bearing claims ({len(weak_lb)}):")
        for wlb in weak_lb:
            print(f"  {wlb[0]} ({wlb[1]}) supports {wlb[2]} ({wlb[3]})")
    print(f"\nWritten: {yaml_path}")
    print(f"Written: {dot_path}")


if __name__ == "__main__":
    main()
