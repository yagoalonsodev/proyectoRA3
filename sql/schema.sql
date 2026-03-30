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

-- Normalización de outcomes (para analizar probabilidad/precio por outcome)
create table if not exists polymarket.dim_outcome (
  outcome_id text primary key,
  market_id text not null references polymarket.dim_market(market_id),
  outcome_index int not null,
  outcome_label text,
  unique (market_id, outcome_index)
);

create index if not exists ix_dim_outcome_market
  on polymarket.dim_outcome (market_id);

create table if not exists polymarket.fact_outcome_snapshot (
  snapshot_ts timestamptz not null,
  market_id text not null references polymarket.dim_market(market_id),
  outcome_id text not null references polymarket.dim_outcome(outcome_id),
  extracted_at text,
  probability numeric,
  raw jsonb,
  primary key (snapshot_ts, market_id, outcome_id)
);

create index if not exists ix_fact_outcome_snapshot_market_ts
  on polymarket.fact_outcome_snapshot (market_id, snapshot_ts desc);
