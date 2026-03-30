import os

import streamlit as st

from src.agent.langgraph_agent import AgentConfig, build_agent


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


st.set_page_config(page_title="Polymarket CSGO Chatbot", layout="centered")
st.title("Chatbot Polymarket (CSGO) · LangGraph + Neon")

neon_url = _env("NEON_DATABASE_URL")
openai_key = _env("OPENAI_API_KEY")
model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

if not neon_url:
    st.error("Falta `NEON_DATABASE_URL` en el entorno.")
    st.stop()
if not openai_key:
    st.error("Falta `OPENAI_API_KEY` en el entorno.")
    st.stop()

agent = build_agent(AgentConfig(neon_database_url=neon_url, openai_api_key=openai_key, openai_model=model))

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask"):
    question = st.text_input(
        "Pregunta",
        placeholder="Ej: ¿Qué mercados han tenido mayor volumen en las últimas 24 horas?",
    )
    show_sql = st.checkbox("Mostrar SQL generado", value=True)
    submitted = st.form_submit_button("Enviar")

if submitted and question.strip():
    state = {"question": question.strip()}
    out = agent.invoke(state)
    st.session_state.history.append(out)

for i, item in enumerate(reversed(st.session_state.history), start=1):
    st.markdown(f"### Consulta #{len(st.session_state.history) - i + 1}")
    st.write(item.get("answer", ""))
    if show_sql:
        st.code(item.get("sql", ""), language="sql")
    rows = item.get("rows", [])
    if rows:
        st.dataframe(rows, use_container_width=True)

