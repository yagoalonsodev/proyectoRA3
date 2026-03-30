import json
import os
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.agent.langgraph_agent import AgentConfig, build_agent

# Las 3 preguntas del enunciado (Parte 2 / demo), siempre disponibles.
DEFAULT_SAVED_QUERIES: list[str] = [
    "¿Qué mercados han tenido mayor volumen esta semana?",
    "¿Qué mercado cambió más de probabilidad en las últimas 24 horas?",
    "¿Qué mercados son los más activos actualmente?",
]

_SAVE_FILE = Path(__file__).resolve().parent / "saved_queries_user.json"


def _load_user_saved() -> list[str]:
    if not _SAVE_FILE.exists():
        return []
    try:
        data = json.loads(_SAVE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def _save_user_saved(queries: list[str]) -> None:
    _SAVE_FILE.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")


def _merged_saved_queries(user_extra: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in DEFAULT_SAVED_QUERIES + user_extra:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


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

if "history" not in st.session_state:
    st.session_state.history = []
if "user_saved_queries" not in st.session_state:
    st.session_state.user_saved_queries = _load_user_saved()
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
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

with st.sidebar:
    st.markdown("### Consultas guardadas")
    st.caption("Incluye las 3 del enunciado; las que guardes se guardan en este equipo.")
    all_q = _merged_saved_queries(st.session_state.user_saved_queries)
    for i, q in enumerate(all_q):
        label = q if len(q) <= 56 else q[:53] + "…"
        if st.button(label, key=f"load_saved_{i}", use_container_width=True):
            st.session_state.question_input = q
            st.rerun()

with st.form("ask"):
    question = st.text_input(
        "Pregunta",
        placeholder="Ej: ¿Qué mercados han tenido mayor volumen en las últimas 24 horas?",
        key="question_input",
    )
    show_sql = st.checkbox("Mostrar SQL generado", value=True)
    st.caption("Bonus: si preguntas por noticias, usa HLTV (CSGO/CS2).")
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.form_submit_button("Enviar", use_container_width=True)
    with col2:
        save_from_form = st.form_submit_button("Guardar esta consulta", use_container_width=True)

q_text = (question or "").strip()

if save_from_form and q_text:
    merged = _merged_saved_queries(st.session_state.user_saved_queries)
    if q_text not in merged:
        st.session_state.user_saved_queries.append(q_text)
        _save_user_saved(st.session_state.user_saved_queries)
        st.success("Consulta guardada (aparece en la barra lateral).")
    else:
        st.info("Esa consulta ya está en la lista.")

if submitted and q_text:
    state = {"question": q_text}
    with st.spinner("Pensando… (Ollama puede tardar un poco)"):
        try:
            out = agent.invoke(state)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error ejecutando el agente: {e}")
            out = {"question": q_text, "error": str(e)}
    if out is None:
        out = {"question": q_text, "error": "El agente devolvió None."}
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
