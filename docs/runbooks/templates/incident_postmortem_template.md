# Incident Postmortem Template (EMP Encyclopedia Appendix F Alignment)

> **Purpose:** Provide a structured, repeatable format for antifragile
> operations reviews. Fill within 24 hours of incident close and link the
> completed document in the Ops Command dashboard.

## 1. Executive Summary
- **Incident ID:** <!-- e.g. 2025-01-17_FIX_ORDER_STALL -->
- **Severity:** <!-- Critical | High | Medium | Low -->
- **Start Time (UTC):**
- **End Time (UTC):**
- **Detected By:** <!-- Sensor, alert, operator -->
- **Primary Owner:**
- **Status:** <!-- Resolved | Monitoring | Follow-up Required -->
- **High-Level Summary:** <!-- 1-2 sentences describing symptom & resolution -->

## 2. Impact Assessment
- **Systems Affected:** <!-- Trading runtime, ingest, risk engine, etc. -->
- **Client / Capital Impact:** <!-- Quantify PnL, orders blocked, exposure -->
- **SLO Breach:** <!-- Yes/No, reference SLO metrics -->
- **Encyclopedia Cross-Refs:** <!-- e.g. Appendix F §2.1 (Incident Taxonomy) -->

## 3. Timeline of Events
| Timestamp (UTC) | Actor | Event Description |
|-----------------|-------|-------------------|
| <!-- 2025-01-17T10:02:11Z --> | <!-- FIX monitor --> | <!-- Alert fired for missing fills --> |
|  |  |  |

> Include detection, escalations, mitigation attempts, decisions, and recovery.
> Annotate whether the step was automated, manual, or hybrid.

## 4. Root Cause Analysis (RCA)
1. **Symptom:** <!-- Observable effect -->
2. **Proximate Cause:** <!-- Immediate trigger -->
3. **Contributing Factors:** <!-- Latency, config drift, vendor outage -->
4. **Systemic Root Cause:** <!-- Map to Encyclopedia Appendix F failure modes -->
5. **Why Chain:** <!-- 3-5 Whys leading to systemic cause -->

## 5. Mitigation & Recovery
- **Immediate Actions Taken:** <!-- Fixes applied during incident -->
- **Risk Mitigations Triggered:** <!-- Circuit breakers, hedges, kill-switch -->
- **State Validation:** <!-- Reconciliation scripts, order lifecycle dry-run -->
- **Residual Risk:** <!-- Remaining exposure and monitoring plan -->

## 6. Preventative Actions & Follow-Up
| Action Item | Owner | Due Date | Status | Notes |
|-------------|-------|----------|--------|-------|
| <!-- Harden FIX heartbeat thresholds --> | <!-- ops@ --> | <!-- 2025-01-24 --> | <!-- In Progress --> |  |

- **Capital Efficiency Memo Link:** <!-- If incident impacted budgets -->
- **Documentation Updates:** <!-- Runbooks, encyclopedia chapters, config -->
- **Automation Opportunities:** <!-- Tests, alerts, dashboards -->

## 7. Communications
- **Stakeholder Notifications:** <!-- Clients, leadership, vendors -->
- **Public Disclosure:** <!-- Blog, status page, regulatory filings -->
- **Postmortem Review Date:** <!-- When the team will review the document -->

## 8. Attachments & Evidence
- **Order Event Journal Snapshot:** <!-- Path to data_foundation/events/... -->
- **Position Reconciliation Report:** <!-- Path to scripts/reconcile_positions output -->
- **Latency Metrics Export:** <!-- Grafana / Prometheus query link -->
- **Additional Artefacts:** <!-- Logs, dashboards, GA results -->

## 9. Lessons Learned
- **What worked well:** <!-- Processes/tools that reduced impact -->
- **What needs improvement:** <!-- Gaps found in detection or response -->
- **Encyclopedia Alignment:** <!-- Reinstate antifragile principles strengthened -->

---

*Template version 1.0 — maintained in `/docs/runbooks/templates/`. Submit
improvements via PR referencing the relevant roadmap workstream.*
