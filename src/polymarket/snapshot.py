from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SnapshotMeta:
    extracted_at_utc: str
    extracted_at_epoch_ms: int
    dt: str
    hour: int


def snapshot_meta(extracted_at: datetime | None = None) -> SnapshotMeta:
    ts = extracted_at or datetime.now(timezone.utc)
    extracted_at_utc = ts.isoformat().replace("+00:00", "Z")
    extracted_at_epoch_ms = int(ts.timestamp() * 1000)
    dt = ts.strftime("%Y-%m-%d")
    hour = int(ts.strftime("%H"))
    return SnapshotMeta(
        extracted_at_utc=extracted_at_utc,
        extracted_at_epoch_ms=extracted_at_epoch_ms,
        dt=dt,
        hour=hour,
    )


def add_snapshot_metadata(rows: list[dict[str, Any]], meta: SnapshotMeta) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["_extracted_at"] = meta.extracted_at_utc
        rr["_extracted_at_epoch_ms"] = meta.extracted_at_epoch_ms
        rr["_dt"] = meta.dt
        rr["_hour"] = meta.hour
        out.append(rr)
    return out
