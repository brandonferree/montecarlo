# Becker Capital — Monte Carlo Portfolio Analysis (Streamlit)

A self-contained web tool that lets advisors configure a portfolio Monte Carlo
analysis and download the same Becker-branded PDF report as the original deliverable.

## Files

- `app.py` — the entire application: simulation engine, chart rendering, PDF builder, and Streamlit UI
- `requirements.txt` — pinned dependencies

Everything runs in one process; no database, no separate API.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What the user can configure

- **Portfolio basics:** initial investment, time horizon, inflation rate, distribution frequency (annual / quarterly / monthly)
- **Return assumptions:** preset packages (1960–2024 Becker default, 1930–2024 long history, forward-looking) or fully custom μ/σ for equity and fixed income
- **Scenarios:** 1, 2, or 3 side-by-side scenarios. Each scenario has its own name, equity %, and annual distribution amount — letting you compare allocations *and* withdrawal levels
- **Simulation paths:** 1,000 to 25,000

Live preview shows median Year-N values, 20th/80th-percentile bands, probability of ruin, and a path chart that updates as inputs change. Click **Build PDF Report** to generate the branded six-page PDF.

## Deploy options

- **Streamlit Community Cloud** — free, deploy by connecting a GitHub repo; good for internal demos
- **Render / Railway / Fly.io** — small paid plans, custom domain, fits a private advisor team
- **Internal server** — `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` behind a reverse proxy

For multi-user production, consider:
- Adding basic auth (Streamlit supports it via secrets / OAuth) if the tool is exposed externally
- Caching: simulation results are already cached per input set via `@st.cache_data`
- Logging: pipe `streamlit` stdout to a log aggregator if you need an audit trail

## Architecture notes

- Simulation logic (`simulate_scenario`, `run_all_simulations`) is pure NumPy and has zero coupling to Streamlit — it can be lifted into a FastAPI endpoint later if the tool grows.
- Chart functions return `BytesIO` (no temp files) so the same code works in cloud sandboxes.
- The PDF builder writes to an in-memory buffer and the result is sent directly to the browser via `st.download_button` — nothing is persisted to disk.
