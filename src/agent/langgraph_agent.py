from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sqlalchemy.engine import make_url
from sqlalchemy import create_engine, text
import requests


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
    engine = create_engine(neon_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]

def news_tool(query: str, *, max_records: int = 5) -> list[dict[str, Any]]:
    """Bonus Tool: busca noticias públicas via GDELT (sin API key)."""
    q = (query or "").strip()
    if not q:
        return []
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(int(max_records)),
        "sort": "hybridrel",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() if r.content else {}
    arts = data.get("articles") or []
    out: list[dict[str, Any]] = []
    for a in arts:
        out.append(
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "sourceCountry": a.get("sourceCountry"),
                "sourceCollection": a.get("sourceCollection"),
                "seendate": a.get("seendate"),
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

    # Asegura compatibilidad con SQLAlchemy: aceptar postgresql+psycopg:// también.
    try:
        u = make_url(cfg.neon_database_url)
        if u.drivername == "postgresql+psycopg":
            cfg = AgentConfig(**{**cfg.__dict__, "neon_database_url": cfg.neon_database_url.replace("postgresql+psycopg://", "postgresql://", 1)})
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

    def news_node(state: AgentState) -> AgentState:
        if not cfg.enable_news_tool:
            return state
        q = str(state.get("question") or "").strip().lower()
        # Activación simple (no bloquea el flujo normal).
        if "noticia" in q or "news" in q or "twitter" in q or "x " in q or " x" in q:
            try:
                state["news"] = news_tool(str(state.get("question") or ""), max_records=5)
            except Exception as e:  # noqa: BLE001
                state["news"] = []
                state["news_error"] = str(e)
        return state

    def exec_node(state: AgentState) -> AgentState:
        sql = str(state.get("sql") or "")
        # Requisito 14: Database Tool obligatoria (aquí se usa).
        rows = database_tool(cfg.neon_database_url, sql)
        state["rows"] = rows
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
    g.add_node("news", news_node)
    g.add_node("exec", exec_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("sql")
    g.add_edge("sql", "news")
    g.add_edge("news", "exec")
    g.add_edge("exec", "answer")
    g.add_edge("answer", END)
    return g.compile()

