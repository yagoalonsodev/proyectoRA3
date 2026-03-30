from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from sqlalchemy import create_engine, text


@dataclass
class AgentConfig:
    neon_database_url: str
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"


class AgentState(dict):
    # keys: question, sql, rows, answer, error
    pass


def run_sql(neon_database_url: str, sql: str) -> list[dict[str, Any]]:
    engine = create_engine(neon_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def build_agent(cfg: AgentConfig):
    llm = ChatOpenAI(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        temperature=0,
    )

    system_sql = (
        "Eres un asistente de datos. Genera UNA consulta SQL válida para Postgres.\n"
        "Usa SOLO estas tablas:\n"
        "- polymarket.dim_market(market_id,title,question,slug,active,closed,archived,created_at,updated_at,raw)\n"
        "- polymarket.fact_market_snapshot(snapshot_ts,market_id,extracted_at,volume,liquidity,best_bid,best_ask,outcome_prices,outcomes,raw)\n"
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

    def exec_node(state: AgentState) -> AgentState:
        sql = str(state.get("sql") or "")
        rows = run_sql(cfg.neon_database_url, sql)
        state["rows"] = rows
        return state

    system_answer = (
        "Eres un asistente. Te doy la pregunta, el SQL ejecutado y las filas devueltas.\n"
        "Responde en español, claro y conciso. Si hay 0 filas, dilo y sugiere otra consulta.\n"
    )

    def answer_node(state: AgentState) -> AgentState:
        q = str(state.get("question") or "")
        sql = str(state.get("sql") or "")
        rows = state.get("rows") or []
        msg = llm.invoke(
            [
                {"role": "system", "content": system_answer},
                {"role": "user", "content": f"Pregunta: {q}\nSQL: {sql}\nFilas: {rows}"},
            ]
        )
        state["answer"] = (msg.content or "").strip()
        return state

    g = StateGraph(AgentState)
    g.add_node("sql", sql_node)
    g.add_node("exec", exec_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("sql")
    g.add_edge("sql", "exec")
    g.add_edge("exec", "answer")
    g.add_edge("answer", END)
    return g.compile()

