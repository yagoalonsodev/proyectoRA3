from __future__ import annotations

import boto3


def write_success_marker(
    *,
    bucket: str,
    key: str,
    aws_region: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
    body: str = "ok\n",
) -> None:
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id or None,
        aws_secret_access_key=aws_secret_access_key or None,
        aws_session_token=aws_session_token or None,
        region_name=aws_region or None,
    )
    s3 = session.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))

