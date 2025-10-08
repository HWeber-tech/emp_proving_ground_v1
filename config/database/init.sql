-- TimescaleDB initialisation for the developer data backbone.
-- The script is executed automatically by the docker entrypoint and ensures the
-- extension is available before ingest jobs run.
CREATE EXTENSION IF NOT EXISTS timescaledb;
