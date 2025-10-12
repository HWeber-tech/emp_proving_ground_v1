# Final Dry Run Review Playbook

This playbook turns the evidence captured during the 72 hour AlphaTrade dry
run into a meeting-ready brief and supporting archive.

1. **Collect evidence**
   - Export structured JSONL logs for the entire run (one file per service or
     day is fine).
   - Export the decision diary (`decision_diary.json`) and performance telemetry
     (`performance_metrics.json`) using the existing tooling in
     `tools/operations/final_dry_run_audit.py` or the runtime supervisor.
2. **Generate review materials**
   - Run the new review CLI to analyse the evidence, produce the wrap-up brief,
     and optionally materialise an evidence packet directory plus archive:

     ```bash
     python tools/operations/final_dry_run_review.py \
       --log path/to/logs/*.jsonl \
       --diary path/to/decision_diary.json \
       --performance path/to/performance_metrics.json \
       --run-label "AlphaTrade Final Dry Run" \
       --attendee "Ops Lead" --attendee "Risk Chair" \
       --sign-off --sign-off-min-duration-hours 72 \
       --packet-dir artifacts/final-dry-run \
       --packet-archive artifacts/final-dry-run.tar.gz \
       --output artifacts/final-dry-run-brief.md
     ```

     The command writes a Markdown agenda that summarises highlights, flags any
     incidents, and appends the detailed audit + sign-off readouts. Exit status
     is non-zero when the review fails or (optionally) warns, making it easy to
     wire into CI.
3. **Hold the review**
   - Circulate the generated brief (and archive if requested) ahead of the
     governance review. The *Action Items* section already lists every log gap,
     diary warning, performance issue, or sign-off finding so the team can agree
     on remediations during the meeting.
   - Capture any additional follow-ups with `--note` or by appending to the
     brief. Re-run the tool as new evidence arrives to keep the agenda current.

This playbook complements the existing `dry_run_audit` flow by adding a
structured wrap-up artefact and optional evidence packet, covering the "Final
Dry Run & Sign-off" roadmap item without modifying the roadmap document itself.
