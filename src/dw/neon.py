from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, create_engine, text


@dataclass(frozen=True)
class NeonConfig:
    database_url: str


def make_engine(cfg: NeonConfig) -> Engine:
    return create_engine(cfg.database_url, pool_pre_ping=True)


def ensure_schema(engine: Engine, *, schema_sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(schema_sql))

def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def _maybe_json_loads(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:  # noqa: BLE001
                return v
    return v


def upsert_market(engine: Engine, *, market: dict[str, Any]) -> None:
    market_id = str(market.get("id") or market.get("marketId") or market.get("market_id") or "")
    if not market_id:
        return

    def _ts(v: Any) -> datetime | None:
        if not v:
            return None
        # La API suele devolver timestamps en string; lo guardamos tal cual si no parsea
        if isinstance(v, datetime):
            return v
        try:
            # Soporta "2026-..." o epoch
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(float(v), tz=timezone.utc)
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    payload = {
        "market_id": market_id,
        "title": market.get("title"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "archived": market.get("archived"),
        "created_at": _ts(market.get("createdAt") or market.get("created_at")),
        "updated_at": _ts(market.get("updatedAt") or market.get("updated_at")),
        "raw": _json_dumps(market),
    }

    sql = text(
        """
        insert into polymarket.dim_market (
          market_id, title, question, slug, active, closed, archived, created_at, updated_at, raw
        )
        values (
          :market_id, :title, :question, :slug, :active, :closed, :archived, :created_at, :updated_at,
          cast(:raw as jsonb)
        )
        on conflict (market_id) do update set
          title = excluded.title,
          question = excluded.question,
          slug = excluded.slug,
          active = excluded.active,
          closed = excluded.closed,
          archived = excluded.archived,
          created_at = coalesce(excluded.created_at, polymarket.dim_market.created_at),
          updated_at = coalesce(excluded.updated_at, polymarket.dim_market.updated_at),
          raw = excluded.raw
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, payload)


def insert_snapshot(engine: Engine, *, snapshot_ts: datetime, market: dict[str, Any]) -> None:
    market_id = str(market.get("id") or market.get("marketId") or market.get("market_id") or "")
    if not market_id:
        return

    # muchos campos pueden venir como strings numéricos
    def _num(v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    payload = {
        "snapshot_ts": snapshot_ts,
        "market_id": market_id,
        "extracted_at": market.get("_extracted_at"),
        "volume": _num(market.get("volume")),
        "liquidity": _num(market.get("liquidity")),
        "best_bid": _num(market.get("bestBid") or market.get("best_bid")),
        "best_ask": _num(market.get("bestAsk") or market.get("best_ask")),
        "outcome_prices": _json_dumps(_maybe_json_loads(market.get("outcomePrices") or market.get("outcome_prices"))),
        "outcomes": _json_dumps(_maybe_json_loads(market.get("outcomes"))),
        "raw": _json_dumps(market),
    }

    sql = text(
        """
        insert into polymarket.fact_market_snapshot (
          snapshot_ts, market_id, extracted_at, volume, liquidity, best_bid, best_ask,
          outcome_prices, outcomes, raw
        )
        values (
          :snapshot_ts, :market_id, :extracted_at, :volume, :liquidity, :best_bid, :best_ask,
          cast(:outcome_prices as jsonb), cast(:outcomes as jsonb), cast(:raw as jsonb)
        )
        on conflict (snapshot_ts, market_id) do update set
          extracted_at = excluded.extracted_at,
          volume = excluded.volume,
          liquidity = excluded.liquidity,
          best_bid = excluded.best_bid,
          best_ask = excluded.best_ask,
          outcome_prices = excluded.outcome_prices,
          outcomes = excluded.outcomes,
          raw = excluded.raw
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, payload)


def upsert_outcome(
    engine: Engine,
    *,
    market_id: str,
    outcome_index: int,
    outcome_label: str | None,
) -> str:
    outcome_id = f"{market_id}:{outcome_index}"
    payload = {
        "outcome_id": outcome_id,
        "market_id": market_id,
        "outcome_index": outcome_index,
        "outcome_label": outcome_label,
    }
    sql = text(
        """
        insert into polymarket.dim_outcome (outcome_id, market_id, outcome_index, outcome_label)
        values (:outcome_id, :market_id, :outcome_index, :outcome_label)
        on conflict (outcome_id) do update set
          outcome_label = excluded.outcome_label
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, payload)
    return outcome_id


def insert_outcome_snapshot(
    engine: Engine,
    *,
    snapshot_ts: datetime,
    market_id: str,
    outcome_id: str,
    extracted_at: str | None,
    probability: float | None,
    raw: dict[str, Any],
) -> None:
    payload = {
        "snapshot_ts": snapshot_ts,
        "market_id": market_id,
        "outcome_id": outcome_id,
        "extracted_at": extracted_at,
        "probability": probability,
        "raw": _json_dumps(raw),
    }
    sql = text(
        """
        insert into polymarket.fact_outcome_snapshot (
          snapshot_ts, market_id, outcome_id, extracted_at, probability, raw
        )
        values (
          :snapshot_ts, :market_id, :outcome_id, :extracted_at, :probability, cast(:raw as jsonb)
        )
        on conflict (snapshot_ts, market_id, outcome_id) do update set
          extracted_at = excluded.extracted_at,
          probability = excluded.probability,
          raw = excluded.raw
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, payload)

