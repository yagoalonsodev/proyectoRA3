# Proyecto Final: Data Engineering + AI Agents con Polymarket (CSGO)

Este proyecto implementa:

- Pipeline de datos orquestado con **Apache Airflow**
- **DataLake en S3** con datos **RAW** en **Delta Lake**
- **Sensor en Airflow** que espera la llegada del RAW antes de transformar
- Transformaciones y carga a **Data Warehouse relacional (NeonDB/Postgres)**
- **Agente LangGraph** + **Chatbot Streamlit** para consultas en lenguaje natural (SQL contra Neon)

## Requisitos (alto nivel)

- Docker (para Airflow)
- Credenciales AWS (S3 real) y un bucket
- Un proyecto en NeonDB (Postgres) y su `NEON_DATABASE_URL`
- Clave de OpenAI si vas a usar el agente con `langchain-openai`

## Setup rápido

1. Copia variables:

```bash
cp .env.example .env
```

2. Crea el esquema del data warehouse en Neon (una vez por proyecto; si no, la base queda sin tablas y el pipeline no puede cargar datos):

```bash
pip install 'psycopg[binary]'   # si aún no lo tienes
python scripts/init_neon_schema.py
```

El script lee `NEON_DATABASE_URL` desde `.env` y ejecuta `sql/schema.sql` (schema `polymarket` + tablas).

3. Arranca Airflow:

```bash
docker compose up -d
```

4. UI Airflow: `http://localhost:8080`

- Usuario: `yago`
- Password: `yago`

## Ejecutar el chatbot (Streamlit)

Desde tu máquina (no dentro de Airflow):

```bash
export $(cat .env | xargs)  # o carga variables a tu manera
streamlit run streamlit_app/app.py
```

## Notas importantes

- `config/` está ignorado por git (contiene contraseñas). No lo subas a GitHub.
- El sensor del pipeline usa el objeto `_SUCCESS` en S3 para garantizar que la transformación solo corre cuando el RAW está disponible.
- **Cada ejecución exitosa del DAG** (cada hora, con `catchup=False` solo la ventana programada) sigue la cadena del enunciado: extracción → RAW en S3 → sensor → **transformación y carga al Data Warehouse (Neon)**. La tarea `transform_and_load_to_neon` es la que materializa los snapshots en Postgres en cada corrida.

## Estructura

- `dags/`: DAG del pipeline
- `src/`: librería Python (cliente Polymarket, S3/Delta, transformaciones, carga a Neon)
- `sql/`: DDL del DW
- `streamlit_app/`: chatbot (LangGraph + Streamlit)
- `docs/`: arquitectura y guía de demo
