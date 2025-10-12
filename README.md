
## Mini-cycle orchestration quickstart

EMP exposes lightweight orchestration entry points for the day 1–2 and day 3–4
mini-cycles.  Use the CLI to trigger either phase from a developer workstation:

```bash
emp mini-cycle            # runs the Lion/FlashAttention pipeline
emp mini-cycle --days d3d4  # runs the retrieval-memory A/B harness
```

The day 3–4 runner wires the retrieval-memory store (FAISS + SQLite) behind a
flag so it can be enabled for inference-only evaluation.  It automatically
persists the fitted regime model, enforces configurable retention limits on the
memory index, captures latency overhead for the feature hook, and publishes the
regime-repeat decision report alongside CSV and markdown summaries under
`artifacts/reports/mc_d3d4/`.
