from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from deltalake import DeltaTable


@dataclass(frozen=True)
class RawPartition:
    dt: str
    hour: int


def read_raw_markets_partition(
    *,
    delta_table_uri: str,
    partition: RawPartition,
    aws_region: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
) -> pd.DataFrame:
    storage_options: dict[str, str] = {}
    if aws_region:
        storage_options["AWS_REGION"] = aws_region
    if aws_access_key_id:
        storage_options["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if aws_secret_access_key:
        storage_options["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if aws_session_token:
        storage_options["AWS_SESSION_TOKEN"] = aws_session_token

    dt = DeltaTable(delta_table_uri, storage_options=storage_options or None)
    df = dt.to_pandas(partitions=[("_dt", "=", partition.dt), ("_hour", "=", partition.hour)])
    return df


def snapshot_timestamp_from_partition(partition: RawPartition) -> datetime:
    # snapshot_ts = inicio de hora UTC
    return datetime.fromisoformat(f"{partition.dt}T{partition.hour:02d}:00:00+00:00").astimezone(
        timezone.utc
    )


def rows_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")

