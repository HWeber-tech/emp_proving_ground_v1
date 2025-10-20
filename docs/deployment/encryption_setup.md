# Encryption deployment playbook

This runbook explains how to provision, store, and validate the encryption
artifacts required by the Professional Predator runtime. It aligns the roadmap
security milestone with concrete deployment steps so TLS, encrypted storage, and
telemetry remain reproducible across environments.

## Scope and prerequisites

- A certificate authority or managed TLS service (for example, AWS ACM) for
  ingress-facing endpoints; self-signed assets are acceptable in non-production.
- A secret store (Kubernetes SealedSecrets, Vault, SSM Parameter Store, or the
  locked-down dotenv workflow in `docs/operations/env_security_hardening.md`).
- Operators with access to the runtime hosts to mount certificates and manage
  encrypted volumes.

## Secret and certificate handling

1. Keep PEM bundles (`*.crt`, `*.key`, optional `ca.crt`) alongside other
   runtime secrets under `/etc/emp/secrets` or the equivalent sealed secret.
   The directory hardening guidance applies to certificates as well as API
   credentials.
2. Track certificate metadata (issuer, expiry) in the deployment ticket. Renew
   at least 14 days before expiry and re-issue sealed secrets or host files in
   place.
3. For Kubernetes clusters, store runtime TLS material in a SealedSecret and
   mount it as a projected volume. Docker Compose deployments should mount the
   host directory read-only:
   ```yaml
   volumes:
     - type: bind
       source: /etc/emp/secrets/runtime_tls
       target: /run/emp/tls
       read_only: true
   ```
4. Copy the certificate chain into the trust store when private PKI roots are
   used: `sudo cp ca.crt /usr/local/share/ca-certificates/emp-runtime.crt &&
   sudo update-ca-certificates`.

## Generating TLS assets for staging

Use the snippet below to create temporary certificates for test clusters:

```bash
mkdir -p /etc/emp/secrets/runtime_tls
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout /etc/emp/secrets/runtime_tls/runtime.key \
  -out /etc/emp/secrets/runtime_tls/runtime.crt \
  -subj "/CN=runtime.local" -addext "subjectAltName=DNS:runtime.local,DNS:localhost"
chmod 640 /etc/emp/secrets/runtime_tls/runtime.*
```

Record the SANs you include; clients such as Prometheus or the load balancer
must resolve one of them.

## In-transit encryption wiring

### HTTP ingress and service endpoints

- AWS ingress overlays already require TLS 1.2+ via the ALB annotations (see
  `k8s/aws-production-setup.yaml:271`). Replace the placeholder ACM certificate
  ARN with the real asset before rollout.
- Internal runtime health endpoints run behind `RuntimeHealthServer`, which
  enforces TLS by loading the provided certificate/key pair and refusing to
  start when they are missing (`src/runtime/healthcheck.py:570-588`). Mount the
  certificate volume and point the runtime extras at the same paths used by the
  metrics exporter (see below) so the server can boot.

### Metrics exporter

- Set `METRICS_EXPORTER_TLS_CERT_PATH` / `METRICS_EXPORTER_TLS_KEY_PATH` in the
  SystemConfig extras (or `EMP_METRICS_TLS_CERT_PATH` / `EMP_METRICS_TLS_KEY_PATH`
  via the environment) before starting the runtime. The exporter refuses to
  launch without TLS material and logs the bound port on success
  (`main.py:128-140`, `src/operational/metrics.py:886-903,928`).
- Provide the same CA bundle to the scraping stack (for example, by mounting
  it on the Prometheus pod).

### TimescaleDB / PostgreSQL

`TimescaleConnectionSettings` automatically threads through the standard libpq
options (`TIMESCALEDB_SSLMODE`, `TIMESCALEDB_SSLROOTCERT`, `TIMESCALEDB_SSLCERT`,
`TIMESCALEDB_SSLKEY`) when they are present (`src/data_foundation/persist/timescale.py:282-311`).
Set at least:

- `TIMESCALEDB_SSLMODE=require`
- `TIMESCALEDB_SSLROOTCERT=/run/emp/tls/ca.crt`

Use `sslmode=verify-full` and provide client certs when the database instance
requires mutual TLS.

### Redis cache

`RedisConnectionSettings` treats `rediss://` URLs or `REDIS_SSL=true` as a
signal to enable TLS (`src/data_foundation/cache/redis_cache.py:128-150`).
Update the connection secrets accordingly and run
`redis-cli --tls --key runtime.key --cert runtime.crt --cacert ca.crt -u
rediss://…` during deployment validation.

### Kafka ingest publishers

The Kafka connector switches to encrypted connections when credentials are
present or when `KAFKA_SECURITY_PROTOCOL` is set to `SASL_SSL`
(`src/data_foundation/streaming/kafka_stream.py:1107-1145`). Configure:

- `KAFKA_SECURITY_PROTOCOL=SASL_SSL`
- `KAFKA_SASL_MECHANISM=PLAIN` (or SCRAM as required)
- `KAFKA_USERNAME` / `KAFKA_PASSWORD`
- Optional `KAFKA_SSL_CA_LOCATION` when your brokers use a private CA

Store these entries in the same sealed secret used for the runtime extras.

## At-rest encryption

### DuckDB artefacts

Enable encrypted volumes for local DuckDB storage with the built-in helper:

- Set `EMP_DUCKDB_ENCRYPTED_ROOT=/mnt/emp-encrypted` to force the runtime to
  relocate database files onto the encrypted mount.
- Add `EMP_REQUIRE_ENCRYPTED_DUCKDB=1` in production so boot fails if the mount
  is absent (`src/data_foundation/duckdb_security.py:17-78`).
- Owners should provide an encrypted filesystem (LUKS, dm-crypt, or a cloud
  volume with storage-level encryption) at the configured root.

### Managed databases and backups

- Enable storage-level encryption on the Timescale/PostgreSQL service (for
  example, AWS RDS with KMS keys) and document the key policy alongside the
  runbook entry.
- Ensure backup tooling writes to encrypted object storage. Capture the storage
  target in `TIMESCALE_BACKUP_STORAGE` so backup telemetry reflects the control
  surface (`docs/operations/backup_plan.md`).

## Deployment checklist

1. Regenerate or validate TLS certificates; update sealed secrets or host
   mounts.
2. Set the Timescale, Redis, Kafka, and metrics TLS environment variables in
   the release manifest (`config/deployment/*.yaml` or the Docker/Kubernetes
   overlays).
3. Confirm DuckDB encrypted root mounts are present on runtime hosts before
   starting the process.
4. Run smoke tests:
   - `curl -vk https://runtime.example.com/health` (expect certificate chain and
     401 without a bearer token).
   - `openssl s_client -connect host:8081 -CAfile ca.crt` for the metrics port.
   - Application-level checks for Redis (`redis-cli --tls`) and Kafka (`kcat
     -X security.protocol=SASL_SSL …`).
5. Feed observed TLS versions into the security telemetry by exporting
   `SECURITY_TLS_VERSIONS="TLS1.2,TLS1.3"` and `SECURITY_LEGACY_TLS_IN_USE=0` so
   the security posture snapshot reports a pass (`src/operations/security.py:112-170`).

## Ongoing verification

- Monitor the runtime logs for the `🔐 Security posture snapshot` and
  `Prometheus metrics exporter started with TLS…` lines; unexpected downgrades
  require immediate investigation (`src/runtime/runtime_builder.py`,
  `src/operational/metrics.py:928`).
- Add certificate expiry alerts to the observability stack; expose the expiry
  timestamp via custom metrics or configuration snapshots if your platform
  supports it.
- During quarterly drills, replay the CLI smoke tests above and confirm backups
  remain encrypted end-to-end.
