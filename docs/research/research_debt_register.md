# Research Debt Register

## Purpose
Track outstanding research questions, data gaps, and follow-up experiments so
that exploratory work remains visible and actionable. The register is reviewed
during the first Friday of each month by the research council.

## Review protocol
1. Update status fields before the meeting using evidence from experiment logs
   and telemetry reports.
2. Mark items as **Resolved** only when acceptance evidence is stored in
   `artifacts/research/` and referenced in the relevant briefs.
3. Escalate blockers to the delivery roadmap and assign owners when dependencies
   extend beyond the research team.

## Register

| ID | Theme | Question / hypothesis | Current status | Next action | Owner | Target review |
| --- | --- | --- | --- | --- | --- | --- |
| RD-001 | Evolution | Does species-aware selection preserve alpha during volatility spikes? | Investigating | Run speciation backtest with March 2020 data; log telemetry to `artifacts/evolution/live_runs/`. | Evolution lead | 2025-02-07 |
| RD-002 | Sentiment | Are transformer-based sentiment models stable across European languages? | Pending data | Acquire multilingual corpora; evaluate zero-shot performance; update NLP roadmap. | NLP squad | 2025-03-07 |
| RD-003 | Causal ML | How sensitive are treatment effects to liquidity regime changes? | In progress | Perform sensitivity analysis using volatility regime classifier output; document in causal metrics brief. | Quant research | 2025-02-07 |
| RD-004 | Compliance | What audit artefacts are required for MiFID II RTS 6 alignment? | Discovery | Interview compliance advisors; extend `docs/operations/regulatory_telemetry.md`. | Compliance liaison | 2025-02-07 |
| RD-005 | Operations | Can cross-region ingest failover meet sub-5 minute RTO? | Blocked | Awaiting infrastructure provisioning; capture requirement in OPS-310 epic. | Platform team | 2025-03-07 |

## Action log
- 2025-01-10: Initial register created alongside high-impact roadmap refresh.
- 2025-01-24: Added RD-003 following causal inference spike outcomes.
- 2025-02-07: Schedule to revisit RD-001, RD-003, and RD-004 after February
  telemetry drills.

## Meeting cadence
- Chair rotates monthly between research leads.
- Minutes stored under `docs/research/meetings/<YYYY-MM-DD>.md`.
- Decisions feed directly into roadmap checkboxes and epic grooming.
