#!/usr/bin/env python
"""
LNN + PyReason Bidirectional Integration POC: Trust Network

Demonstrates a feedback loop where:
  - LNN reasons locally:  expert + endorsed -> reliable -> credible
  - PyReason propagates through graph:  credible + trusts -> endorsed
  - Feedback tightens bounds each round, cascading credibility hop-by-hop

Neither system alone achieves the combined result:
  - LNN can't traverse graph edges (no trusts propagation)
  - PyReason can't do local logical composition (no expert+endorsed -> reliable)
"""

import sys
import os

import networkx as nx
from lnn import (
    Model,
    Predicate,
    Variable,
    And,
    Implies,
    Forall,
    Fact as LNNFact,
    World,
)
import pyreason as pr
from pyreason.scripts.learning.classification.lnn_classifier import (
    LNNClassifier,
    LNNInterfaceOptions,
)

NODES = ["alice", "bob", "charlie"]

# Minimum lower bound to inject a fact into PyReason.
# Bounds near [0, 1] are essentially UNKNOWN and would not trigger rules usefully.
INJECT_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_knowledge_graph():
    """Create the trust network: alice --trusts--> bob --trusts--> charlie."""
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edge("alice", "bob", trusts=1)
    g.add_edge("bob", "charlie", trusts=1)
    return g


def build_lnn_model():
    """Create the LNN model with expert/endorsed/reliable/credible predicates."""
    model = Model()
    x = Variable("x")

    expert = Predicate("expert")
    endorsed = Predicate("endorsed")
    reliable = Predicate("reliable")
    credible = Predicate("credible")

    # Rules as axioms (enables bidirectional inference)
    rule1 = Forall(x, Implies(And(expert(x), endorsed(x)), reliable(x)))
    rule2 = Forall(x, Implies(And(reliable(x), expert(x)), credible(x)))
    model.add_knowledge(rule1, rule2, world=World.AXIOM)

    predicate_map = {
        "expert": expert,
        "endorsed": endorsed,
        "reliable": reliable,
        "credible": credible,
    }
    return model, predicate_map


def get_initial_data():
    """Initial LNN evidence."""
    return {
        "expert": {
            "alice": LNNFact.TRUE,
            "bob": (0.7, 0.9),
            "charlie": (0.6, 0.8),
        },
        "reliable": {"alice": (0.8, 1.0)},
        "endorsed": {"alice": LNNFact.TRUE},
    }


def setup_pyreason(graph):
    """Reset PyReason and load the graph + propagation rule."""
    pr.reset()
    pr.reset_rules()
    pr.reset_settings()
    pr.settings.verbose = False
    pr.load_graph(graph)
    # Explicit [0,1] on credible clause so the rule fires for any non-trivial
    # credible bound, not just TRUE.  Without it the parser defaults to [1,1].
    pr.add_rule(
        pr.Rule("endorsed(y) <- credible(x):[0,1], trusts(x,y)", "trust_propagation")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lnn_bounds_for(predicate_map, pred_name):
    """Extract {grounding_str: (lower, upper)} for a predicate."""
    pred = predicate_map[pred_name]
    bounds = {}
    for grounding in pred.groundings:
        name = grounding[0] if isinstance(grounding, tuple) and len(grounding) == 1 else grounding
        tensor = pred.get_data(grounding)
        bounds[name] = (tensor[0, 0].item(), tensor[0, 1].item())
    return bounds


def print_lnn_bounds(predicate_map, predicates=None):
    """Pretty-print LNN bounds for the given (or all) predicates."""
    if predicates is None:
        predicates = ["expert", "endorsed", "reliable", "credible"]
    for pred_name in predicates:
        bounds = lnn_bounds_for(predicate_map, pred_name)
        if not bounds:
            continue
        entries = ", ".join(
            f"{n}=[{l:.4f},{u:.4f}]" for n, (l, u) in sorted(bounds.items())
        )
        print(f"    {pred_name}: {entries}")


def extract_pyreason_bounds(interpretation, predicates, nodes):
    """Pull bounds from PyReason's final-timestep interpretation dict."""
    results = {}
    interp_dict = interpretation.get_dict()
    max_t = max(interp_dict.keys())
    final = interp_dict[max_t]

    for pred in predicates:
        results[pred] = {}
        for node in nodes:
            if node in final and pred in final[node]:
                lower, upper = final[node][pred]
                results[pred][node] = (float(lower), float(upper))
    return results


def inject_credible_facts(target_bounds, t1=0, t2=2):
    """Inject credible bounds into PyReason, skipping near-UNKNOWN entries."""
    credible = target_bounds.get("credible", {})
    injected = []
    for node in NODES:
        if node not in credible:
            continue
        lower, upper = credible[node]
        if lower <= INJECT_THRESHOLD:
            continue
        fact = pr.Fact(
            f"credible({node}):[{lower},{upper}]",
            f"lnn-credible-{node}",
            start_time=t1,
            end_time=t2,
        )
        pr.add_fact(fact)
        injected.append((node, lower, upper))
    return injected


# ---------------------------------------------------------------------------
# Scenario A: LNN only
# ---------------------------------------------------------------------------

def run_lnn_only(initial_data):
    print("=" * 60)
    print("  BASELINE A: LNN Only (no graph propagation)")
    print("=" * 60)

    model, predicate_map = build_lnn_model()

    formatted = {predicate_map[k]: v for k, v in initial_data.items()}
    model.add_data(formatted)
    model.infer()

    print("\n  All LNN bounds after inference:")
    print_lnn_bounds(predicate_map)

    print("\n  -> LNN computes credible(alice) from reliable+expert,")
    print("     but bob/charlie lack endorsed data so their credible")
    print("     bounds remain wide (close to UNKNOWN).\n")


# ---------------------------------------------------------------------------
# Scenario B: PyReason only (seeded with LNN's initial credible facts)
# ---------------------------------------------------------------------------

def run_pyreason_only(initial_data, graph):
    print("=" * 60)
    print("  BASELINE B: PyReason Only (single LNN seed, one round)")
    print("=" * 60)

    # Run LNN once to produce credible facts
    model, predicate_map = build_lnn_model()
    formatted = {predicate_map[k]: v for k, v in initial_data.items()}
    model.add_data(formatted)
    model.infer()

    credible_bounds = lnn_bounds_for(predicate_map, "credible")
    print("\n  LNN credible bounds (seed):")
    for node in NODES:
        if node in credible_bounds:
            l, u = credible_bounds[node]
            tag = " <- injected" if l > INJECT_THRESHOLD else " <- skipped (near UNKNOWN)"
            print(f"    credible({node}): [{l:.4f}, {u:.4f}]{tag}")

    # Setup PyReason and inject credible facts
    setup_pyreason(graph)
    # Build target dict in the shape inject_credible_facts expects
    inject_credible_facts({"credible": credible_bounds})

    interpretation = pr.reason(timesteps=2)

    bounds = extract_pyreason_bounds(interpretation, ["endorsed", "credible"], NODES)
    print("\n  PyReason results (final timestep):")
    for pred in ["endorsed", "credible"]:
        for node in NODES:
            if node in bounds.get(pred, {}):
                l, u = bounds[pred][node]
                print(f"    {pred}({node}): [{l:.4f}, {u:.4f}]")

    print("\n  -> PyReason propagates endorsed(bob) from credible(alice)+trusts,")
    print("     but it cannot compute credible(bob) because it lacks the LNN")
    print("     rule expert+endorsed -> reliable -> credible.\n")


# ---------------------------------------------------------------------------
# Scenario C: Combined bidirectional feedback loop
# ---------------------------------------------------------------------------

def run_combined(initial_data, graph):
    print("=" * 60)
    print("  COMBINED: Bidirectional LNN <-> PyReason Feedback Loop")
    print("=" * 60)

    model, predicate_map = build_lnn_model()

    classifier = LNNClassifier(
        lnn_model=model,
        predicate_map=predicate_map,
        target_predicates=["credible"],
        identifier="lnn",
        interface_options=LNNInterfaceOptions(
            convergence_threshold=0.001,
            max_feedback_rounds=5,
            bound_tightening_only=True,
        ),
    )

    max_rounds = 5
    convergence_threshold = 0.001
    prev_credible = {}

    for round_num in range(1, max_rounds + 1):
        print(f"\n  {'─' * 50}")
        print(f"  Round {round_num}")
        print(f"  {'─' * 50}")

        # --- LNN forward ---
        if round_num == 1:
            raw, target, facts = classifier.forward(data=initial_data, t1=0, t2=2)
        else:
            raw, target, facts = classifier.forward(t1=0, t2=2)

        print("\n  LNN bounds:")
        for pred_name in ["endorsed", "reliable", "credible"]:
            if pred_name not in raw:
                continue
            entries = []
            for node in NODES:
                if node in raw[pred_name]:
                    l, u = raw[pred_name][node]
                    entries.append(f"{node}=[{l:.4f},{u:.4f}]")
            if entries:
                print(f"    {pred_name}: {', '.join(entries)}")

        # --- Inject credible facts into PyReason ---
        setup_pyreason(graph)
        injected = inject_credible_facts(target)

        print(f"\n  Injected {len(injected)} credible fact(s) into PyReason:")
        for node, l, u in injected:
            print(f"    credible({node}): [{l:.4f}, {u:.4f}]")
        if not injected:
            print("    (none with meaningful lower bounds)")

        # --- PyReason reasoning ---
        interpretation = pr.reason(timesteps=2)

        pyreason_bounds = extract_pyreason_bounds(interpretation, ["endorsed"], NODES)
        endorsed_bounds = pyreason_bounds.get("endorsed", {})

        print(f"\n  PyReason endorsed results:")
        if endorsed_bounds:
            for node in NODES:
                if node in endorsed_bounds:
                    l, u = endorsed_bounds[node]
                    print(f"    endorsed({node}): [{l:.4f}, {u:.4f}]")
        else:
            print("    (no new endorsed bounds)")

        # --- Check convergence on credible bounds ---
        current_credible = target.get("credible", {})
        max_change = 0.0
        for node in NODES:
            if node in current_credible:
                l, u = current_credible[node]
                if node in prev_credible:
                    old_l, old_u = prev_credible[node]
                    change = max(abs(l - old_l), abs(u - old_u))
                    max_change = max(max_change, change)
                else:
                    max_change = max(max_change, max(abs(l), abs(u)))

        prev_credible = dict(current_credible)

        if round_num > 1 and max_change < convergence_threshold:
            print(f"\n  Converged! (max credible change: {max_change:.6f})")
            break

        # --- Feed endorsed bounds back to LNN ---
        if endorsed_bounds:
            classifier.receive_feedback({"endorsed": endorsed_bounds})
            print(f"\n  -> Fed endorsed bounds back to LNN for next round")

    # Final summary table
    print(f"\n  {'─' * 50}")
    print("  Final credible bounds:")
    for node in NODES:
        if node in prev_credible:
            l, u = prev_credible[node]
            print(f"    credible({node}): [{l:.4f}, {u:.4f}]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  LNN + PyReason Bidirectional Integration POC")
    print("  Trust Network: alice --trusts--> bob --trusts--> charlie")
    print("=" * 60)
    print()
    print("  LNN rules (local reasoning):")
    print("    expert(x) AND endorsed(x) -> reliable(x)")
    print("    reliable(x) AND expert(x) -> credible(x)")
    print()
    print("  PyReason rule (graph propagation):")
    print("    credible(x), trusts(x,y) -> endorsed(y)")
    print()
    print("  Initial data:")
    print("    expert:   alice=TRUE, bob=[0.7,0.9], charlie=[0.6,0.8]")
    print("    reliable: alice=[0.8,1.0]")
    print("    endorsed: alice=TRUE")
    print()

    graph = build_knowledge_graph()
    initial_data = get_initial_data()

    run_lnn_only(initial_data)
    run_pyreason_only(initial_data, graph)
    run_combined(initial_data, graph)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  LNN alone:      Computes credible(alice), but bob/charlie stay")
    print("                   near UNKNOWN (no graph traversal capability).")
    print("  PyReason alone:  Propagates endorsed(bob) via trust edge, but")
    print("                   can't compute credible(bob) (lacks LNN rules).")
    print("  Combined:        Each round discovers one more hop:")
    print("                   R1: credible(alice) -> endorsed(bob)")
    print("                   R2: credible(bob)   -> endorsed(charlie)")
    print("                   R3: credible(charlie) computed -> converge")
    print("                   Bounds attenuate with graph distance.")
    print()


if __name__ == "__main__":
    main()
