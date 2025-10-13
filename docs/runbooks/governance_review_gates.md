# Governance Review Gates Playbook

This playbook captures how to track AlphaTrade governance review gates, record
verdicts, and publish evidence for sign-off meetings. It complements the policy
ledger and promotion tooling by covering the roadmap requirement for recorded
review gates with explicit sign-off criteria.

## Inspect gate status

Run the status command to merge the static definitions under
`config/governance/review_gates.yaml` with the latest gate state and print a
JSON or Markdown summary:

```bash
python -m tools.governance.review_gates status \
  --definitions config/governance/review_gates.yaml \
  --state artifacts/governance/review_gates.json
```

Use `--format markdown` to emit a Markdown table suitable for governance packs
and `--workflow-output` to persist a compliance workflow snapshot for the
operational evidence archive:

```bash
python -m tools.governance.review_gates status \
  --format markdown \
  --workflow-output artifacts/governance/review_gates_workflow.json
```

## Record a verdict

During the sign-off review, capture the decision with the `decide` subcommand.
Specify each mandatory criterion as `met`, `not_met`, or `waived` to keep the
record aligned with the gate definition:

```bash
python -m tools.governance.review_gates decide \
  --gate operations_final_signoff \
  --verdict pass \
  --decided-by "Ops Lead" \
  --decided-by "Risk Chair" \
  --criterion dry_run_duration=met \
  --criterion evidence_packet=met \
  --criterion signoff_verdict=met \
  --persist artifacts/governance/review_gates.json
```

The command updates `artifacts/governance/review_gates.json` atomically and
prints a JSON payload summarising the new verdict so meeting notes and runbooks
can reference the same source of truth.

## Gate definitions

The default definitions cover the four governance gates highlighted in the
roadmap: TRM governance alignment, perception live feed sign-off, adaptation
promotion readiness, and the final dry-run sign-off. Update
`config/governance/review_gates.yaml` to add or refine criteria when governance
requirements change. Tests under `tests/governance/test_review_gates.py` and
`tests/tools/test_review_gates_cli.py` guard the contract so changes remain
regression-tested.
