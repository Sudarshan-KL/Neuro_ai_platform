# Frontend UI

Production frontend for the Neuro-AI platform.

## Run locally

From the project root:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

- UI: `http://127.0.0.1:8000/`
- Static assets: `http://127.0.0.1:8000/ui/`
- Backend docs: `http://127.0.0.1:8000/docs`

The frontend is served by FastAPI and reads live telemetry from:

- `GET /api/v1/health`
- `GET /api/v1/models`
- `GET /api/v1/alerts`
- `POST /api/v1/predict_window`
- `POST /api/v1/upload_edf`
