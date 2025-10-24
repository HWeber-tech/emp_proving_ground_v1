# Tick Data Schema

## Purpose

Institutional ingest now records every trade, quote, and depth sample in a
canonical layout so downstream sensors, replay engines, and analytics can read a
single contract regardless of source venue. The schema targets TimescaleDB in
production with a SQLite-compatible dialect for test fixtures, and is optimised
for the access patterns exercised by `TimescaleQueryInterface`.

## Canonical Types

| Model | Description |
| --- | --- |
| `TradeTick` | Single execution event with sequencing metadata, venue tags, and encoded trade conditions. |
| `QuoteTick` | Normalised top-of-book bid/ask snapshot with derived mid-price and spread basis-points. |
| `OrderBookSnapshot` + `OrderBookLevel` | Depth snapshot capturing ladder levels, liquidity imbalance, and sequencing metadata. |

All symbols are upper-cased, venues are trimmed, and optional enumerations
(`conditions`, `liquidity_side`) are serialised as tuples. Negative prices or
sizes are rejected at validation time. Levels in a snapshot are deduplicated and
sorted.

## Timescale Storage Layout

| Table | Primary Key | Columns | Notes |
| --- | --- | --- | --- |
| `market_data.ticks` | `(symbol, ts, sequence)` | `price`, `size`, `venue`, `trade_id`, `liquidity_side`, `conditions`, `source`, `ingested_at` | Sequence auto-fills per `(symbol, ts)` when upstream connectors omit it. `conditions` stored as JSON array for extensibility. |
| `market_data.quotes` | `(symbol, ts, sequence)` | `bid_price`, `bid_size`, `ask_price`, `ask_size`, `mid_price`, `spread_bps`, `venue`, `source`, `ingested_at` | Hypertable + descending `(symbol, ts)` index match historical replay queries and resampling routines. |
| `market_data.order_book` | `(symbol, ts, sequence, level)` | `bid_price`, `bid_size`, `ask_price`, `ask_size`, `imbalance`, `venue`, `source`, `ingested_at` | Each row represents a single ladder level; imbalance is derived when absent. |

All tables are promoted to hypertables when running against PostgreSQL /
TimescaleDB and receive a descending `(symbol, ts)` index to support recent
window scans. SQLite deployments receive identically named tables (prefixed with
schema name) for deterministic testing.

## Ingestion Expectations

`TimescaleIngestor` exposes three new upsert helpers:

- `upsert_ticks(df, source="unknown")`
- `upsert_quotes(df, source="unknown")`
- `upsert_order_book(df, source="unknown")`

Each helper accepts a pandas frame that may omit `sequence`—the ingestor assigns
stable counters per `(symbol, ts)` group before writing. Optional arrays (trade
conditions) are JSON-serialised, and datetime fields are normalised to UTC. The
upserts are idempotent: replays overwrite existing rows by key, keeping
`ingested_at` fresh for provenance.

## Access Pattern Alignment

`TimescaleQueryInterface` already issues projections against `ticks`, `quotes`,
and `order_book`. With the schema in place, its default column selections (`ts,
price/size, ingested_at`, etc.) are now backed by materialised tables rather
than test-only fixtures. Query caching continues to operate on immutable result
frames.

## Retention & Governance

- Tick hypertables inherit Timescale’s compression policies once activated. The
  schema keeps sequencing metadata so rollups can recreate execution order even
  after aggregation.
- `source` and optional `venue` tags enable partitioned purges by provider when
  contracts require data removal.
- `ingested_at` timestamps enforce observability SLAs and allow staleness checks
  to surface in the operations dashboard.

## Next Steps

1. Wire live market adapters to emit the canonical pydantic models so validation
   failures are caught before persistence.
2. Add retention/compression policies in migrations once Timescale is deployed
   beyond the SQLite harness.
3. Extend replay utilities to stream from the new tables instead of fixture
   parquet/JSONL files.
