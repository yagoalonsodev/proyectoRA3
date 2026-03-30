import os
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.agent.langgraph_agent import AgentConfig, build_agent


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env, override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

st.set_page_config(page_title="Polymarket CSGO Chatbot", layout="centered")
st.title("Chatbot Polymarket (CSGO) · LangGraph + Neon")

neon_url = _env("NEON_DATABASE_URL")
llm_provider = (_env("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
openai_key = _env("OPENAI_API_KEY")
openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
ollama_base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
ollama_model = _env("OLLAMA_MODEL", "deepseek-coder:6.7b") or "deepseek-coder:6.7b"

if not neon_url:
    st.error("Falta `NEON_DATABASE_URL` en el entorno.")
    st.stop()

if llm_provider == "openai" and not openai_key:
    st.error("LLM_PROVIDER=openai pero falta `OPENAI_API_KEY` en el entorno.")
    st.stop()

agent = build_agent(
    AgentConfig(
        neon_database_url=neon_url,
        llm_provider=llm_provider,
        openai_api_key=openai_key,
        openai_model=openai_model,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        enable_news_tool=True,
    )
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask"):
    question = st.text_input(
        "Pregunta",
        placeholder="Ej: ¿Qué mercados han tenido mayor volumen en las últimas 24 horas?",
    )
    show_sql = st.checkbox("Mostrar SQL generado", value=True)
    st.caption("Bonus: si preguntas por noticias, usa HLTV (CSGO/CS2).")
    submitted = st.form_submit_button("Enviar")

if submitted and question.strip():
    state = {"question": question.strip()}
    with st.spinner("Pensando… (Ollama puede tardar un poco)"):
        try:
            out = agent.invoke(state)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error ejecutando el agente: {e}")
            out = {"question": question.strip(), "error": str(e)}
    if out is None:
        out = {"question": question.strip(), "error": "El agente devolvió None."}
    st.session_state.history.append(out)

for i, item in enumerate(reversed(st.session_state.history), start=1):
    st.markdown(f"### Consulta #{len(st.session_state.history) - i + 1}")
    if not isinstance(item, dict):
        st.error(f"Entrada inválida en historial: {type(item).__name__}")
        continue
    st.write(item.get("answer", ""))
    if item.get("error"):
        st.error(item.get("error"))
    news = item.get("news") or []
    if news:
        with st.expander("Noticias HLTV (raw)", expanded=False):
            st.json(news)
    if show_sql and not item.get("news_only"):
        st.code(item.get("sql", ""), language="sql")
    rows = item.get("rows", [])
    if rows and not item.get("news_only"):
        st.dataframe(rows, use_container_width=True)

