#!/usr/bin/env python3
"""
Crea el esquema y tablas del DW en Neon (Postgres) a partir de sql/schema.sql.

Neon ya viene con la base `neondb` creada; lo que falta suele ser el esquema
`polymarket` y las tablas. Sin este paso la base aparece vacía y el pipeline
fallará al hacer upserts.

Uso (desde la raíz del proyecto):

    python scripts/init_neon_schema.py

Lee NEON_DATABASE_URL de .env en la raíz, o de la variable de entorno.
Formato URL: postgresql://... o postgresql+psycopg://...
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_neon_url_from_dotenv(env_path: Path) -> str | None:
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("NEON_DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _normalize_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_sql_statements(sql: str) -> list[str]:
    """Parte el script en sentencias; el DDL del repo no incluye ';' dentro de strings."""
    text = re.sub(r"--[^\n]*", "", sql)
    parts: list[str] = []
    for block in text.split(";"):
        s = block.strip()
        if s:
            parts.append(s + ";")
    return parts


def main() -> int:
    root = _project_root()
    schema_file = root / "sql" / "schema.sql"
    if not schema_file.is_file():
        print(f"No se encuentra {schema_file}", file=sys.stderr)
        return 1

    url = os.environ.get("NEON_DATABASE_URL") or _load_neon_url_from_dotenv(root / ".env")
    if not url:
        print(
            "Define NEON_DATABASE_URL en el entorno o en .env en la raíz del proyecto.",
            file=sys.stderr,
        )
        return 1

    url = _normalize_url(url)
    sql_text = schema_file.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_text)
    if not statements:
        print(f"No hay sentencias en {schema_file}", file=sys.stderr)
        return 1

    try:
        import psycopg
    except ImportError:
        print("Instala psycopg: pip install 'psycopg[binary]'", file=sys.stderr)
        return 1

    print(f"Aplicando {len(statements)} sentencia(s) en Neon…")
    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            for i, stmt in enumerate(statements, 1):
                cur.execute(stmt)
                print(f"  [{i}/{len(statements)}] ok")
    print("Esquema aplicado: schema polymarket y tablas listas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
