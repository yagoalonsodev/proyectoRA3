from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sqlalchemy.engine import make_url
from sqlalchemy import create_engine, text
import requests
import re
import xml.etree.ElementTree as ET


@dataclass
class AgentConfig:
    neon_database_url: str
    llm_provider: str = "ollama"  # "ollama" | "openai"
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    # Bonus tools
    enable_news_tool: bool = True


class AgentState(dict):
    # keys: question, sql, rows, news, answer, error
    pass


def database_tool(neon_database_url: str, sql: str) -> list[dict[str, Any]]:
    """Database Tool (obligatoria): ejecuta SQL contra Neon/Postgres y devuelve filas."""
    # Preferir psycopg3 (evita dependencia de psycopg2).
    url = (neon_database_url or "").strip()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        # Si el LLM devolviese algo que no retorna filas (p.ej. DDL), no rompemos el agente.
        if not getattr(result, "returns_rows", False):
            return []
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]

def _extract_cs_team_terms(rows: list[dict[str, Any]], *, max_terms: int = 8) -> list[str]:
    """
    Extrae términos tipo 'Team A' y 'Team B' desde títulos/preguntas para usarlos en la búsqueda.
    Heurística: split por 'vs', 'v', '-' y limpiar tokens cortos.
    """
    texts: list[str] = []
    for r in rows[:50]:
        for k in ("title", "question"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())

    # Candidatos: partes alrededor de "vs"
    candidates: list[str] = []
    for t in texts:
        # normaliza separadores típicos de partidos
        parts = re.split(r"\s+(?:vs\.?|v\.?)\s+|\s*-\s*", t, flags=re.IGNORECASE)
        for p in parts:
            s = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ ._'-]", " ", p).strip()
            s = re.sub(r"\s{2,}", " ", s)
            if 2 <= len(s) <= 40:
                candidates.append(s)

    # Filtra ruido común
    stop = {
        "csgo",
        "cs2",
        "counter strike",
        "counter-strike",
        "match",
        "map",
        "bo1",
        "bo3",
        "bo5",
        "winner",
        "win",
        "who wins",
        "will",
    }
    terms: list[str] = []
    for c in candidates:
        lc = c.lower()
        if lc in stop:
            continue
        if any(x in lc for x in ("polymarket", "yes", "no", "over", "under")):
            continue
        if lc.isdigit():
            continue
        # evita frases demasiado largas
        if len(lc.split()) > 5:
            continue
        if c not in terms:
            terms.append(c)
        if len(terms) >= max_terms:
            break
    return terms


def _build_csgo_news_query(question: str, rows: list[dict[str, Any]]) -> str:
    # Query base CSGO/CS2 + esports + torneos comunes.
    base = [
        '"Counter-Strike"',
        "CSGO",
        "CS2",
        '"Counter Strike"',
        "HLTV",
        "ESL",
        "BLAST",
        "IEM",
        "PGL",
        "Major",
    ]
    team_terms = _extract_cs_team_terms(rows, max_terms=8)
    # Entrecomilla términos compuestos para mejorar el matching
    quoted = [f'"{t}"' if " " in t else t for t in team_terms]
    q_terms = base + quoted
    # GDELT query: OR entre términos (limitado a contexto CSGO/CS2/torneos/equipos).
    return " OR ".join(q_terms)

def _hltv_rss_news(*, max_records: int = 20) -> list[dict[str, Any]]:
    """
    Fuente de noticias CSGO/CS2 (bonus) sin API key: RSS de HLTV.
    Devuelve items recientes (título + link + fecha).
    """
    url = "https://www.hltv.org/rss/news"
    headers = {
        # HLTV suele requerir UA para devolver contenido correctamente.
        "User-Agent": "Mozilla/5.0 (compatible; ProyectoRA3/1.0; +https://example.invalid)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    # RSS 2.0: <rss><channel><item>...
    items = root.findall("./channel/item")
    out: list[dict[str, Any]] = []
    for it in items[: max(1, int(max_records))]:
        out.append(
            {
                "title": (it.findtext("title") or "").strip(),
                "url": (it.findtext("link") or "").strip(),
                "published": (it.findtext("pubDate") or "").strip(),
                "source": "HLTV",
            }
        )
    return out

def _filter_news_items(items: list[dict[str, Any]], *, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """
    Filtra noticias para priorizar equipos/torneos detectados en los mercados.
    Si no hay términos, devuelve las más recientes.
    """
    terms = _extract_cs_team_terms(rows, max_terms=10)
    # Siempre anclado a CSGO/CS2, pero HLTV ya lo está.
    if not terms:
        return items[:limit]
    pat = re.compile("|".join(re.escape(t) for t in terms if t), re.IGNORECASE)
    hit = [x for x in items if pat.search(x.get("title") or "")]
    return (hit or items)[:limit]

def news_tool(*, question: str, rows: list[dict[str, Any]], max_records: int = 8) -> list[dict[str, Any]]:
    """Bonus Tool: noticias de CSGO/CS2 (equipos/torneos) sin finanzas."""
    # Fuente primaria: HLTV RSS (estable, sin API key).
    try:
        items = _hltv_rss_news(max_records=max(20, int(max_records) * 5))
        return _filter_news_items(items, rows=rows or [], limit=int(max_records))
    except Exception:
        # Fallback (best-effort): GDELT puede rate-limit (429); no es crítica.
        q = _build_csgo_news_query(question or "", rows or [])
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": q,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(int(max_records)),
            "sort": "hybridrel",
            "timespan": "7d",
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 429:
            return []
        r.raise_for_status()
        data = r.json() if r.content else {}
        arts = data.get("articles") or []
        out: list[dict[str, Any]] = []
        for a in arts:
            out.append(
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "published": a.get("seendate"),
                    "source": "GDELT",
                }
            )
        return out


def build_agent(cfg: AgentConfig):
    provider = (cfg.llm_provider or "ollama").strip().lower()
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("Falta openai_api_key para llm_provider=openai")
        llm = ChatOpenAI(api_key=cfg.openai_api_key, model=cfg.openai_model, temperature=0)
    elif provider == "ollama":
        llm = ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0,
        )
    else:
        raise ValueError(f"llm_provider no soportado: {cfg.llm_provider!r}")

    # Asegura compatibilidad con SQLAlchemy usando psycopg3.
    try:
        u = make_url(cfg.neon_database_url)
        if u.drivername == "postgresql":
            cfg = AgentConfig(
                **{
                    **cfg.__dict__,
                    "neon_database_url": cfg.neon_database_url.replace(
                        "postgresql://", "postgresql+psycopg://", 1
                    ),
                }
            )
    except Exception:
        pass

    system_sql = (
        "Eres un asistente de datos. Genera UNA consulta SQL válida para Postgres.\n"
        "Usa SOLO estas tablas:\n"
        "- polymarket.dim_market(market_id,title,question,slug,active,closed,archived,created_at,updated_at,raw)\n"
        "- polymarket.fact_market_snapshot(snapshot_ts,market_id,extracted_at,volume,liquidity,best_bid,best_ask,outcome_prices,outcomes,raw)\n"
        "- polymarket.dim_outcome(outcome_id,market_id,outcome_index,outcome_label)\n"
        "- polymarket.fact_outcome_snapshot(snapshot_ts,market_id,outcome_id,extracted_at,probability,raw)\n"
        "Reglas:\n"
        "- Devuelve únicamente SQL (sin markdown, sin explicaciones).\n"
        "- La consulta DEBE ser de solo lectura: un único SELECT (sin INSERT/UPDATE/DELETE/CREATE/DROP/ALTER).\n"
        "- Limita a 50 filas si no se pide explícitamente otra cosa.\n"
        "- Si la pregunta pide \"últimas 24h\" usa now() - interval '24 hours'.\n"
    )

    def sql_node(state: AgentState) -> AgentState:
        q = str(state.get("question") or "")
        msg = llm.invoke(
            [
                {"role": "system", "content": system_sql},
                {"role": "user", "content": q},
            ]
        )
        state["sql"] = (msg.content or "").strip().rstrip(";") + ";"
        return state

    def exec_node(state: AgentState) -> AgentState:
        sql = str(state.get("sql") or "")
        # Requisito 14: Database Tool obligatoria (aquí se usa).
        rows = database_tool(cfg.neon_database_url, sql)
        state["rows"] = rows
        return state

    def news_node(state: AgentState) -> AgentState:
        if not cfg.enable_news_tool:
            return state
        q = str(state.get("question") or "").strip().lower()
        # Solo si el usuario pide noticias (para no ralentizar el flujo normal).
        if "noticia" in q or "news" in q:
            try:
                state["news"] = news_tool(
                    question=str(state.get("question") or ""),
                    rows=state.get("rows") or [],
                    max_records=8,
                )
            except Exception as e:  # noqa: BLE001
                state["news"] = []
                state["news_error"] = str(e)
        return state

    system_answer = (
        "Eres un asistente. Te doy la pregunta, el SQL ejecutado y las filas devueltas.\n"
        "Responde en español, claro y conciso. Si hay 0 filas, dilo y sugiere otra consulta.\n"
        "Si te paso NEWS (lista de noticias), puedes usarlo como contexto adicional.\n"
    )

    def answer_node(state: AgentState) -> AgentState:
        q = str(state.get("question") or "")
        sql = str(state.get("sql") or "")
        rows = state.get("rows") or []
        news = state.get("news") or []
        msg = llm.invoke(
            [
                {"role": "system", "content": system_answer},
                {"role": "user", "content": f"Pregunta: {q}\nSQL: {sql}\nFilas: {rows}\nNEWS: {news}"},
            ]
        )
        state["answer"] = (msg.content or "").strip()
        return state

    g = StateGraph(AgentState)
    g.add_node("sql", sql_node)
    g.add_node("exec", exec_node)
    g.add_node("news", news_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("sql")
    g.add_edge("sql", "exec")
    g.add_edge("exec", "news")
    g.add_edge("news", "answer")
    g.add_edge("answer", END)
    return g.compile()

