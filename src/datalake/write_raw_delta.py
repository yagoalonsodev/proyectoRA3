from __future__ import annotations

import json
from typing import Any

import pandas as pd
from deltalake import write_deltalake


def _prepare_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for k, v in item.items():
        if isinstance(v, (list, dict)):
            row[k] = json.dumps(v) if v else None
        else:
            row[k] = v
    return row


def _sanitize_for_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().all():
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("string")
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("float64")
            else:
                df[col] = df[col].astype("string")
    return df


def write_raw_markets_delta_to_s3(
    *,
    delta_table_uri: str,
    markets: list[dict[str, Any]],
    aws_region: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
    mode: str = "append",
) -> int:
    if not markets:
        return 0

    rows = [_prepare_row(m) for m in markets]
    df = pd.DataFrame(rows)
    df = _sanitize_for_delta(df)

    storage_options: dict[str, str] = {}
    if aws_region:
        storage_options["AWS_REGION"] = aws_region
    if aws_access_key_id:
        storage_options["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if aws_secret_access_key:
        storage_options["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if aws_session_token:
        storage_options["AWS_SESSION_TOKEN"] = aws_session_token

    write_deltalake(
        delta_table_uri,
        df,
        mode=mode,
        partition_by=["_dt", "_hour"],
        storage_options=storage_options or None,
    )
    return len(df)

