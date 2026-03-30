from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str

    def base_uri(self) -> str:
        p = self.prefix.strip("/")
        return f"s3://{self.bucket}/{p}" if p else f"s3://{self.bucket}"

    def raw_markets_delta_uri(self) -> str:
        return f"{self.base_uri()}/raw/polymarket_markets_delta"

    def raw_markets_success_key(self, *, dt: str, hour: int) -> str:
        p = self.prefix.strip("/")
        base = f"{p}/" if p else ""
        return f"{base}raw/polymarket_markets_delta/dt={dt}/hour={hour:02d}/_SUCCESS"
