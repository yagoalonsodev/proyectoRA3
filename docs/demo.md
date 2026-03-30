## Guion de demo (presentación)

### 1) Airflow (pipeline + sensor)
- Abrir UI `http://localhost:8080`
- DAG: `polymarket_csgo_hourly_pipeline`
- Enseñar:
  - Task `extract_to_s3_raw`
  - Task `wait_for_raw_marker` (sensor)
  - Task `transform_and_load_to_neon`

### 2) DataLake (S3)
- Mostrar en AWS Console o con CLI que existe:
  - Delta table: `s3://$S3_BUCKET/$S3_PREFIX/raw/polymarket_markets_delta`
  - Marker por hora: `.../dt=YYYY-MM-DD/hour=HH/_SUCCESS`

### 3) Data Warehouse (Neon)
Ejemplos de SQL:

```sql
select count(*) from polymarket.dim_market;
```

```sql
select market_id, snapshot_ts, volume, liquidity
from polymarket.fact_market_snapshot
order by snapshot_ts desc
limit 10;
```

### 4) Chatbot (Streamlit + LangGraph)
- Lanzar:

```bash
streamlit run streamlit_app/app.py
```

- Probar preguntas:
  - “¿Qué mercados han tenido mayor volumen en las últimas 24 horas?”
  - “¿Qué mercados son los más activos actualmente?”
  - “¿Qué mercado cambió más de liquidez en las últimas 24 horas?”

Activar “Mostrar SQL generado” en la UI.

