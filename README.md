# LNN + PyReason Bidirectional Integration POC

A proof-of-concept demonstrating a feedback loop between [IBM Logical Neural Networks (LNN)](https://github.com/IBM/LNN) and [PyReason](https://github.com/lab-v2/pyreason). LNN reasons locally about individuals using learned logical rules, PyReason propagates results across a graph, and feedback tightens bounds each round — producing results neither system achieves alone.

## Why

LNN and PyReason both operate on interval-valued truth bounds `[lower, upper]`, making them natural integration partners. But they have complementary blind spots:

| Capability | LNN | PyReason |
|---|---|---|
| Local logical composition (e.g. `expert AND endorsed -> reliable`) | Yes | No |
| Graph-structure propagation (e.g. traverse trust edges) | No | Yes |

The `LNNClassifier` (in the [pyreason repo](https://github.com/lab-v2/pyreason/tree/LNN-integration)) bridges this gap.

## Example: Trust Network

```
alice ──trusts──> bob ──trusts──> charlie
```

**LNN rules (local):**
- `expert(x) AND endorsed(x) -> reliable(x)`
- `reliable(x) AND expert(x) -> credible(x)`

**PyReason rule (graph):**
- `credible(x), trusts(x,y) -> endorsed(y)`

**Initial data:**
- `expert`: alice=TRUE, bob=[0.7, 0.9], charlie=[0.6, 0.8]
- `reliable`: alice=[0.8, 1.0]
- `endorsed`: alice=TRUE

### Feedback loop

| Round | What happens | New result |
|---|---|---|
| 1 | LNN computes credible(alice)=[1.0, 1.0] | PyReason propagates endorsed(bob)=[1.0, 1.0] |
| 2 | LNN receives endorsed(bob), computes credible(bob)=[0.4, 1.0] | PyReason propagates endorsed(charlie)=[1.0, 1.0] |
| 3 | LNN receives endorsed(charlie), computes credible(charlie)=[0.2, 1.0] | Converges |

Credible lower bounds attenuate with graph distance (Lukasiewicz t-norm): **1.0 -> 0.4 -> 0.2**

### Three baselines confirm neither system alone produces this

- **LNN-only:** Computes credible(alice) but bob/charlie remain UNKNOWN (can't traverse graph edges)
- **PyReason-only:** Propagates endorsed(bob) but can't compute credible(bob) (lacks the expert+endorsed->reliable rule)
- **Combined:** Each round discovers one more hop, converging in 4 rounds

## Setup

```bash
# Python >= 3.9 required
pip install git+https://github.com/IBM/LNN
pip install pyreason  # or install from local source
```

## Run

```bash
python trust_network_poc.py
```

First run is slow (~60s) due to numba JIT compilation. Subsequent runs are fast.

## Architecture

```
                    ┌─────────────────────┐
                    │      LNN Model      │
                    │  expert AND endorsed │
                    │    -> reliable       │
                    │  reliable AND expert │
                    │    -> credible       │
                    └─────┬─────▲─────────┘
                          │     │
                  credible│     │endorsed
                   bounds │     │bounds
                          │     │
                    ┌─────▼─────┴─────────┐
                    │    LNNClassifier     │
                    │  forward() ──> Facts │
                    │  receive_feedback()  │
                    └─────┬─────▲─────────┘
                          │     │
              PyReason    │     │  extract
                Facts     │     │  bounds
                          │     │
                    ┌─────▼─────┴─────────┐
                    │      PyReason        │
                    │  credible(x),        │
                    │  trusts(x,y)         │
                    │    -> endorsed(y)    │
                    └─────────────────────┘
```

## Related

- [PyReason LNN-integration branch](https://github.com/lab-v2/pyreason/tree/LNN-integration) — contains `LNNClassifier` and `LNNInterfaceOptions`
- [PR #132](https://github.com/lab-v2/pyreason/pull/132) — adds the classifier to PyReason
