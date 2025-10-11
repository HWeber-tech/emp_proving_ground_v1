# Reflection Intelligence Module API

The Reflection Intelligence Module (RIM) exposes an asynchronous API for producing advisory suggestions derived from Decision Diaries. All payloads adhere to `rim.v1` schemas described in [`interfaces/rim_types.json`](../../interfaces/rim_types.json).

## Versioning & Audit

- Every object includes `schema_version`, `input_hash`, `model_hash`, and `config_hash`.
- `input_hash` and `config_hash` use SHA-256 with canonical JSON hashing (see below).
- `model_hash` is opaque and may represent a Git commit, model artifact hash, or placeholder for shadow runs.
- All timestamp fields are UTC ISO-8601. Runners must normalize any local wall-clock inputs to UTC prior to ingest, and diary windows are computed strictly in UTC.
- Published suggestions are immutable JSONL files written to `artifacts/rim_suggestions/` with filename pattern `rim-suggestions-UTC-<ISO>-<RUN_ID>.jsonl`.
- `RUN_ID` values follow `<yyyyMMddHHmmssZ>-<hostname>-<pid>`. Replays over the same window reuse the original `RUN_ID` for idempotency or annotate a `rerun_of` header field referencing the prior run.
- For each emission we log metadata to `artifacts/rim_logs/` capturing runtime percentiles and suggestion counts (`p50_ms`, `p95_ms`, `windows_processed`, `windows_halted_early_%`, `suggestions_emitted`, `suggestions_dropped_low_confidence`).
- Schema evolution policy: current `schema_version` is `"rim.v1"`; readers must ignore unknown fields; any breaking change promotes to `rim.v2` with a minimum 90-day dual-write period emitting both versions.
- Diary inputs are sourced from `diaries_dir` (default `artifacts/diaries/`) and must match the configured `diary_glob` (`diaries-*.jsonl`).

#### Canonical hashing
`input_hash` = SHA-256 over newline-joined **canonical JSON** of the input lines.

```python
import hashlib, json
def canon(obj):  # canonical, deterministic JSON
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def hash_lines(json_objs):
    payload = "\n".join(canon(o) for o in json_objs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```
All timestamps MUST be UTC ISO-8601 before hashing.

## Public Interfaces

### Submit Diary Window

```http
POST /reflection/rim/input-window
Content-Type: application/json
```

```json
{
  "schema_version": "rim.v1",
  "input_hash": "<sha256>",
  "model_hash": "n/a",
  "config_hash": "<sha256>",
  "window": {
    "start": "2024-03-25T00:00:00Z",
    "end": "2024-03-25T23:59:59Z",
    "minutes": 1440
  },
  "entries": [
    {
      "schema_version": "rim.v1",
      "input_hash": "<sha256>",
      "model_hash": "n/a",
      "config_hash": "<sha256>",
      "timestamp": "2024-03-25T14:25:00Z",
      "instrument": "ESM4",
      "strategy_id": "mean_rev_v2",
      "features_digest": {"zscore": -1.8},
      "belief_state_summary": {"prob_up": 0.62},
      "action": "reduce_weight",
      "risk_flags": ["vol_spike"],
      "pnl": -1500.0,
      "outcome_labels": ["drawdown"]
    }
  ],
  "aggregates": {
    "count": 620,
    "pnl_sum": 12400.5,
    "flag_counts": {"vol_spike": 3}
  }
}
```

### Retrieve Suggestions (Shadow Mode)

```http
GET /reflection/rim/suggestions?window_start=2024-03-25T00:00:00Z
```

Response is a JSON array of `RIMSuggestion` objects as documented below.

```json
[
  {
    "schema_version": "rim.v1",
    "input_hash": "<sha256>",
    "model_hash": "stub-trm-v0",
    "config_hash": "<sha256>",
    "suggestion_id": "rim-20240325-0001",
    "type": "WEIGHT_ADJUST",
    "payload": {
      "strategy_id": "mean_rev_v2",
      "proposed_weight_delta": -0.05,
      "window_minutes": 1440
    },
    "confidence": 0.74,
    "rationale": "Persistent drawdown with elevated volatility flags across last 24h window.",
    "audit_ids": ["diary-20240325-1425"],
    "created_at": "2024-03-25T23:59:59Z"
  }
]
```

## Governance Interoperability

1. RIM publishes JSONL artifacts to `artifacts/rim_suggestions/`.
2. Governance ingestion service watches the directory and enqueues entries into the policy ledger review queue.
3. Reviewers annotate acceptance/override decisions, creating downstream records keyed by `suggestion_id`.
4. Acceptance metrics flow back into telemetry (suggestion acceptance percentage) and optional retraining labels.

> RIM cannot mutate live weights. Only the Governance process may apply suggestions after approval.

## Example JSONL Artifact

```
{"schema_version": "rim.v1", "input_hash": "abc123", "model_hash": "stub-trm-v0", "config_hash": "def456", "suggestion_id": "rim-20240325-0001", "type": "STRATEGY_FLAG", "payload": {"strategy_id": "carry_eur", "reason": "Unusual pnl variance"}, "confidence": 0.68, "rationale": "Variance doubled vs 30d baseline.", "audit_ids": ["diary-20240325-1010"], "created_at": "2024-03-25T23:59:59Z"}
```

## Error Handling

- Invalid schema: 422 response with validation errors; CLI tooling (`tools/rim_validate.py`) assists local debugging.
- Corrupt diary JSONL lines are skipped (counted as `skipped_lines`) while continuing processing; total processed lines are logged alongside the skip count.
- Missing optional fields fall back to documented defaults (e.g., `belief_state_summary.vector` defaults to a length-32 zero vector).
- Diary entries are sorted by timestamp prior to hashing/emission; ties preserve original file order to maintain determinism.
- Governance gate disabled: 503 response; callers must confirm `enable_governance_gate` prior to publishing.
- Kill switch enabled: 204 response with no content; RIM logs message and returns immediately.

## Telemetry

Logged fields include `p50_ms`, `p95_ms`, `windows_processed`, `windows_halted_early_%`, `suggestions_emitted`, and `suggestions_dropped_low_confidence` to highlight runtime drift and suggestion budget consumption.

## CHANGELOG

- **2024-XX-XX:** Initial `rim.v1` release covering DecisionDiaryEntry, RIMInputBatch, and RIMSuggestion contracts.
