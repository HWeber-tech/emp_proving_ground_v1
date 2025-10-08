# Runbook – Kafka ingest offset recovery

The Kafka ingest bridge replays institutional ingest telemetry back onto the
runtime event bus.  When offsets become stuck or messages pile up, use this
procedure to diagnose the lag, commit the appropriate offsets, and verify that
telemetry flows again.

## 1. Detect the incident

1. Review the professional runtime summary for the `event_bus` and `kafka` blocks
   or subscribe to `telemetry.kafka.lag`.  The consumer publishes lag snapshots
   whenever monitoring is enabled, including per-partition offsets and total
   backlog counts.【F:src/data_foundation/streaming/kafka_stream.py†L425-L618】【F:src/data_foundation/streaming/kafka_stream.py†L1983-L2006】
2. Inspect ingest dashboards for missing `telemetry.ingest.*` events or stale
   Kafka topics.  When the consumer pauses, the runtime will continue emitting
   Timescale telemetry but downstream consumers will not see the Kafka mirror.
3. Check runtime logs around the ingest bridge; `_handle_message` records
   deserialization errors and `_commit` warns when commits fail so you can
   distinguish serialization issues from offset problems.【F:src/data_foundation/streaming/kafka_stream.py†L1884-L1955】

## 2. Validate configuration

1. Confirm that the ingest consumer remains enabled and subscribed to the correct
   topics.  `create_ingest_event_consumer` accepts manual topic overrides,
   configurable group IDs, and offset reset policies; ensure the deployment still
   references the intended topic set.【F:src/data_foundation/streaming/kafka_stream.py†L2010-L2154】
2. Review commit settings.  If `KAFKA_INGEST_CONSUMER_AUTO_COMMIT` is disabled,
   the runbook assumes `KAFKA_INGEST_CONSUMER_COMMIT_ON_PUBLISH` drives manual
   commits after each message.  Double-check whether asynchronous commits are
   enabled via `KAFKA_INGEST_CONSUMER_COMMIT_ASYNC`; this affects how the
   `_commit` method interacts with the client.【F:src/data_foundation/streaming/kafka_stream.py†L1944-L1981】
3. If lag publishing is disabled, temporarily set `KAFKA_INGEST_CONSUMER_PUBLISH_LAG`
   to `true` and redeploy so you can monitor recovery progress in real time.【F:src/data_foundation/streaming/kafka_stream.py†L1983-L2006】

## 3. Recover the offsets

1. Pause ingest producers or scale down consumers if possible to reduce traffic
   while working the backlog.
2. Use the Kafka CLI or admin client to inspect the consumer group offsets for
   the configured group ID (default `emp-ingest-bridge`).  Compare them with the
   lag snapshot emitted by `capture_consumer_lag` to determine how far behind the
   bridge is.【F:src/data_foundation/streaming/kafka_stream.py†L425-L618】
3. If offsets are stuck, resume the runtime bridge and watch `_commit` logging.
   The helper iterates through common commit signatures and logs when the client
   rejects them, making it clear whether a configuration mismatch or broker error
   is blocking progress.【F:src/data_foundation/streaming/kafka_stream.py†L1884-L1955】
4. When manual intervention is required, use the Kafka CLI to advance the group
   offsets to the last processed message (consult the ingest journal or telemetry
   payloads for the highest confirmed timestamp).  Keep a note of the offset for
   the incident report.

## 4. Validate recovery

1. Resume normal ingest processing and confirm that lag snapshots trend back
   toward zero.  The bridge publishes telemetry on the configured interval, so a
   steady decline indicates success.【F:src/data_foundation/streaming/kafka_stream.py†L1983-L2006】
2. Confirm that `telemetry.ingest`, `telemetry.ingest.health`, and related topics
   resume flowing through the event bus and Kafka publishers.  Runtime logs should
   stop emitting commit warnings once offsets advance.【F:src/data_foundation/streaming/kafka_stream.py†L1702-L1778】【F:src/data_foundation/streaming/kafka_stream.py†L1884-L1955】
3. Update the institutional data backbone alignment brief or on-call notes with
   the root cause, offset details, and any configuration adjustments performed
   during the recovery.
