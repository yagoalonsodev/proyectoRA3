-- Neon (Postgres) Data Warehouse schema

create schema if not exists polymarket;

create table if not exists polymarket.dim_market (
  market_id text primary key,
  title text,
  question text,
  slug text,
  active boolean,
  closed boolean,
  archived boolean,
  created_at timestamptz,
  updated_at timestamptz,
  raw jsonb
);

create table if not exists polymarket.fact_market_snapshot (
  snapshot_ts timestamptz not null,
  market_id text not null references polymarket.dim_market(market_id),
  extracted_at text,
  volume numeric,
  liquidity numeric,
  best_bid numeric,
  best_ask numeric,
  outcome_prices jsonb,
  outcomes jsonb,
  raw jsonb,
  primary key (snapshot_ts, market_id)
);

create index if not exists ix_fact_market_snapshot_market_ts
  on polymarket.fact_market_snapshot (market_id, snapshot_ts desc);
