from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class PolymarketClient:
    base_url: str = "https://gamma-api.polymarket.com"
    rate_limit_delay_s: float = 0.2
    timeout_s: int = 30
    max_retries: int = 3
    retry_delay_s: float = 2.0

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "User-Agent": "ProyectoFinal-PolymarketPipeline/1.0",
            "Accept": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_s)
                else:
                    raise
            finally:
                time.sleep(self.rate_limit_delay_s)
        raise RuntimeError(f"Request failed: {last_exc}")

    def fetch_paginated(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        params = dict(params or {})
        params["limit"] = limit
        params.setdefault("offset", 0)

        all_rows: list[dict[str, Any]] = []
        offset = 0
        detected_limit: int | None = None

        while True:
            params["offset"] = offset
            payload = self._get(endpoint, params=params)

            if isinstance(payload, list):
                batch = payload
            elif isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
                batch = payload["data"]
            elif isinstance(payload, dict):
                batch = [payload]
            else:
                batch = []

            batch_size = len(batch)
            if batch_size == 0:
                break

            # API suele devolver dicts
            all_rows.extend([r for r in batch if isinstance(r, dict)])

            if detected_limit is None and batch_size < int(params["limit"]):
                detected_limit = batch_size

            offset += batch_size
            expected = detected_limit if detected_limit else limit
            if batch_size < expected:
                break

        return all_rows

    def fetch_active_markets(self, *, limit: int = 500) -> list[dict[str, Any]]:
        # En la Gamma API, /markets suele aceptar filtros. Si alguno no aplica, la API los ignora.
        return self.fetch_paginated(
            "markets",
            params={
                "active": "true",
                "closed": "false",
                "archived": "false",
            },
            limit=limit,
        )


def filter_markets_by_keywords(
    markets: list[dict[str, Any]],
    *,
    keywords: list[str],
) -> list[dict[str, Any]]:
    kws = [k.strip().lower() for k in keywords if k.strip()]
    if not kws:
        return markets

    def haystack(m: dict[str, Any]) -> str:
        title = str(m.get("title") or "")
        question = str(m.get("question") or "")
        slug = str(m.get("slug") or "")
        # tags/outcomes a veces vienen serializados; no forzamos parse aquí
        return " ".join([title, question, slug]).lower()

    out: list[dict[str, Any]] = []
    for m in markets:
        h = haystack(m)
        if any(k in h for k in kws):
            out.append(m)
    return out
