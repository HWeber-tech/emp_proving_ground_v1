# Observability Quickstart

This runbook outlines how to forward runtime telemetry into a local
OpenTelemetry collector and inspect structured log events alongside traces.

## Prerequisites

* Install the OpenTelemetry Collector (`otelcol`) locally.
* Ensure Python dependencies include `opentelemetry-sdk` and the OTLP HTTP
  exporter (already bundled in `requirements/base.txt`).

## Start the Collector

```bash
otelcol --config=config/observability/otel-collector.yaml
```

The reference configuration listens on `0.0.0.0:4318` and mirrors telemetry via
its `logging` exporter so you can confirm payload shape from the collector's
stdout while developing.

## Enable Runtime Instrumentation

Export the relevant environment variables before launching the runtime:

```bash
export OTEL_ENABLED=true
export OTEL_SERVICE_NAME=emp-professional-runtime
export OTEL_ENVIRONMENT=local-dev
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4318/v1/logs
```

`configure_structlog` automatically attaches an OTLP handler when
instrumentation is enabled, so every JSON log record is forwarded to the
collector in addition to stdout. The runtime reuses the same configuration for
tracing so spans and log events share correlation identifiers out of the box.

### Shortcut via observability profiles

Instead of exporting individual `OTEL_*` variables you can point the runtime at
`config/observability/logging.yaml` (or a custom profile) using the
`STRUCTLOG_OTEL_CONFIG` extra:

```bash
export STRUCTLOG_OTEL_CONFIG=default  # resolves to config/observability/logging.yaml
python main.py
```

The profile loader translates the YAML schema used by the observability
runbooks into the runtime's OpenTelemetry settings. Resource attributes defined
in the profile (for example, `service.name` and `deployment.environment`) are
propagated automatically so downstream dashboards receive consistent metadata.

## Verifying Delivery

1. Launch `python main.py` in a separate terminal once the collector is running.
2. Watch the collector console for incoming logs and spans. You should see
   structured payloads that mirror the JSON emitted to stdout.
3. Adjust `config/observability/otel-collector.yaml` or add new exporter files
   in the same directory when routing telemetry to alternative backends (e.g.
   Tempo, Loki, or OpenSearch).

## Cleanup

Stop the collector (`Ctrl+C`) and unset the `OTEL_*` environment variables when
finished to revert to local-only logging.

## Related Runbooks

* [Sensor Drift Monitoring](sensor_drift_monitoring.md) — evaluates sensory
  windows for statistical drift and can be wired into CI using
  `scripts/check_sensor_drift.py`.
