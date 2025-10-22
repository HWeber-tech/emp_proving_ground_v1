# Emergency procedures handbook

This handbook equips operators with a repeatable process for handling high-impact incidents that threaten trading continuity, customer data, or regulatory compliance. Use it alongside the existing runbooks and reliability policies to ensure every emergency follows a disciplined, auditable response path.

## Scope and activation

Activate the emergency procedures when any of the following conditions are met:

| Trigger | Examples | Initial ownership |
| --- | --- | --- |
| **Safety or regulatory risk** | Trading halt triggers, fat-finger protections disabled, suspected market abuse. | Incident commander (IC) + Compliance lead |
| **Severe service outage** | Ingest pipeline hard down, risk engine unavailable, widespread API 5xx. | IC + Technical lead |
| **Data integrity compromise** | Corrupted market data, missing reconciliations, unverified model outputs. | IC + Data steward |
| **Security exposure** | Unauthorized access, credential leak, privilege escalation. | IC + Security lead |

If there is uncertainty, err on the side of activating the procedures and downgrading later during triage.

## Response team structure

Assign the following roles at activation time; one person may fill multiple roles in very small incidents but avoid overloading the IC:

- **Incident commander:** Owns the timeline, decision log, and go/no-go calls. Has authority to invoke the [Master Switch Guide](../MASTER_SWITCH_GUIDE.md) when risk thresholds are exceeded.
- **Technical lead:** Coordinates diagnostic and remediation workstreams across engineering pods, referencing targeted runbooks such as [kafka_ingest_offset_recovery](runbooks/kafka_ingest_offset_recovery.md) and [redis_cache_outage](runbooks/redis_cache_outage.md).
- **Communications lead:** Manages stakeholder updates using the templates below, tracks commitments, and loops in compliance or client-facing teams as needed.
- **Scribe:** Captures actions, evidence, and timestamps to feed the post-incident review and audit artefacts.
- **Domain specialists (optional):** Pull in data foundation, risk, or infrastructure SMEs when the incident touches their controls.

## Golden hour checklist

Complete these actions within the first 60 minutes:

1. **Stabilize operations**
   - Trigger emergency switch-off or throttling if safety limits are exceeded (see [Master Switch Guide](../MASTER_SWITCH_GUIDE.md)).
   - Freeze deployments via the promotion gate (see [promotion_gate policy](promotion_gate.md)).
2. **Assess blast radius**
   - Confirm which systems are impacted using observability dashboards ([observability plan](observability_plan.md), [Prometheus and Grafana overview](prometheus_grafana.md)).
   - Validate data quality guardrails and ingest freshness reports.
3. **Establish communications**
   - Create an incident channel (`#inc-{date}-{short-description}`) and populate the timeline template.
   - Send the initial stakeholder alert (template below) within 15 minutes of activation.
4. **Assign owners**
   - Verify each workstream (containment, investigation, stakeholder comms, compliance) has a named lead.
   - Ensure hand-offs are recorded in the log if roles rotate.

## Response phases

The IC should keep the team aligned on the current phase and the exit criteria for each:

1. **Detection & verification:** Validate the triggering alert and rule out false positives. Engage domain SMEs if the signal is ambiguous.
2. **Containment:** Prevent further damage by isolating affected components, disabling automation, or applying temporary risk limits.
3. **Mitigation & recovery:** Execute the relevant runbooks, restore normal service, and monitor telemetry for stability. Document mitigation evidence (commands, dashboards, screenshots).
4. **Communication & coordination:** Maintain 30-minute status updates (or faster if customer impact is severe). Track stakeholder questions and commitments.
5. **Closure criteria:** Declare the incident resolved when impact has ceased, monitoring shows stability for at least one control interval, and stakeholders have been informed of the resolution.
6. **Post-incident review:** Schedule the review within two business days. Summarize contributing factors, corrective actions, and verification plans. File follow-up tasks in the reliability backlog.

## Communication templates

**Initial stakeholder alert**

```
Subject: [EMERGENCY] <system/area> incident activated

Summary: <2 sentence description>
Start time: <UTC timestamp>
Current impact: <customer/system impact>
Actions underway: <containment + investigation>
Next update: <timestamp + channel>
Incident commander: <name/contact>
```

**Status update (every 30 minutes)**

```
Status: <stabilizing / investigating / mitigated>
Key developments: <bullet list>
Risks/blockers: <bullet list>
Help needed: <specific asks>
ETA to next update: <timestamp>
```

**Closure notice**

```
Incident resolved at <UTC timestamp>.
Impact summary: <final narrative>
Follow-up actions: <tickets / owners / due dates>
Review scheduled: <date + facilitator>
```

## Tooling and artefact capture

- **Observability:** Use the Grafana dashboards, Timescale readiness reports, and incident detection system telemetry to gather hard evidence for the timeline.
- **Ticketing:** File remediation work in the reliability backlog and tag with the incident identifier for traceability.
- **Evidence store:** Upload logs, configuration snapshots, and screenshots to the incident folder within the operational knowledge base. Include exports generated by `build_configuration_backup()` when configuration drift is suspected.
- **Audit trail:** Ensure the scribe maintains a chronological log covering commands run, system state changes, and communication timestamps to support the audit program outlined in `docs/audits`.

## Post-incident follow-up

1. **Debrief:** Hold the review within two business days. Invite all responders plus owners of long-term fixes.
2. **Root cause & contributing factors:** Apply a five-whys analysis and map findings to control objectives (risk, compliance, resilience).
3. **Action tracking:** Create remediation issues with clear acceptance criteria and deadlines. Update the [runbook library](runbooks/README.md) if new patterns emerge.
4. **Communication recap:** Provide stakeholders with the final summary, highlighting lessons learned and improvements.
5. **Readiness check:** Validate that monitoring, alerting, and disaster recovery drills still meet policy thresholds after remediation.

Keep this handbook versioned with the repository so that updates can be reviewed alongside other operational controls.
