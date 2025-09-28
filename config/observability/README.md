# OpenTelemetry Collector Configuration

This directory stores reference configuration for a local OpenTelemetry
collector that ingests the runtime's structured logs and spans via the OTLP
HTTP endpoint. The `otel-collector.yaml` profile exposes `0.0.0.0:4318`, batches
incoming telemetry, and mirrors both traces and logs to the collector's own
stdout using the `logging` exporter so engineers can verify payload shape while
iterating locally.

To run the collector with the reference configuration:

```bash
otelcol --config=config/observability/otel-collector.yaml
```

When the collector is running, enable the runtime instrumentation by exporting
matching environment variables before launching `main.py`:

```bash
export OTEL_ENABLED=true
export OTEL_SERVICE_NAME=emp-professional-runtime
export OTEL_ENVIRONMENT=local-dev
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4318/v1/logs
```

The same endpoint accepts traces, so the trading runtime sends both structured
log events and span data through the collector without additional setup. Update
or extend this configuration when forwarding telemetry to other backends (for
example, Tempo, Loki, or OpenSearch) by adding the relevant exporters under
`config/observability/`.
