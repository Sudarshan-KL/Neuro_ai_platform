# Run And Deploy Guide

This guide explains how to run the project locally, how to use Docker, and how to deploy the app to the internet.

## Local Run

### 1. Install Python dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

### 2. Watch out for the local `DEBUG` environment variable

On this machine, a global `DEBUG=release` environment value caused Pydantic settings parsing issues. If you see startup errors related to `DEBUG`, run commands like this:

```powershell
$env:DEBUG='false'
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Start the app

```powershell
$env:DEBUG='false'
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Open the app

- UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`

## What Runs Where

The app is now unified:

- FastAPI serves the API.
- FastAPI also serves the frontend from `/`.
- Static frontend assets are exposed under `/ui`.

That means the production deployment path is a single service.

## Local Test Files

Use these compatible EDFs for current deployment testing:

- [background_A_Z001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_A_Z001.edf)
- [background_B_O001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_B_O001.edf)
- [seizure_E_S001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/seizure_E_S001.edf)

Do not use the CHB-MIT EDFs with the currently active model unless you first load a matching multi-channel checkpoint.

## Docker Run

### Build and start

```powershell
docker-compose up --build
```

### Exposed port

- App: `http://127.0.0.1:8000/`

### Volumes already configured

The Docker Compose setup mounts:

- `./data`
- `./saved_models`
- `./logs`

That means model artifacts and runtime logs stay visible outside the container.

## Recommended Production Environment Variables

These are the main settings to manage in deployment:

```env
DEBUG=false
LOG_LEVEL=INFO
MODEL_VERSION=v1
SEIZURE_THRESHOLD=0.5
```

If you deploy a different checkpoint later, update `MODEL_VERSION` accordingly.

## Deploy To The Internet

There are three practical ways to deploy this project.

### Option 1. Deploy with Docker on a cloud VM

This is the most flexible option.

Recommended providers:

- AWS EC2
- Azure VM
- Google Compute Engine
- DigitalOcean Droplet
- Hetzner Cloud

Basic flow:

1. Create a Linux VM.
2. Install Docker and Docker Compose.
3. Copy the repo to the server.
4. Place the desired model checkpoint in `saved_models/`.
5. Run:

```bash
docker-compose up --build -d
```

6. Put Nginx or Caddy in front of the app for TLS and domain routing.
7. Point your domain DNS to the server IP.

Recommended production setup:

- App on internal port `8000`
- Reverse proxy on `80/443`
- HTTPS enabled
- Firewall open only for `80` and `443`

### Option 2. Deploy directly with Uvicorn behind Nginx

Use this if you do not want Docker.

Basic flow:

1. Provision a Linux VM.
2. Install Python 3.11, system libs, and Nginx.
3. Create a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app with a process manager such as `systemd`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

6. Put Nginx in front of it for TLS and domain routing.

### Option 3. Container platform deployment

You can also deploy the app to platforms that support Docker containers, such as:

- Render
- Railway
- Fly.io
- Azure Container Apps
- Google Cloud Run

Important note:

- The application expects a local checkpoint under `saved_models/`.
- Some platforms have ephemeral filesystems, so you must either bake the model into the image or mount persistent storage.

## Recommended Internet Deployment Pattern

For this specific project, the best deployment shape is:

1. Docker container for the FastAPI app
2. Mounted or baked-in `saved_models/`
3. Reverse proxy with HTTPS
4. Health check hitting `/api/v1/health`
5. Locked-down CORS origins for the real domain

## Minimum Production Hardening Checklist

Before exposing the app publicly, do these:

- Set `DEBUG=false`
- Restrict `CORS_ORIGINS` to your real domain
- Run behind HTTPS
- Add authentication if the app is not meant to be public
- Persist logs
- Persist alerts and uploads if operational history matters
- Add request size limits for uploads
- Monitor CPU and memory usage during EDF uploads

## Example Nginx Reverse Proxy

Example idea:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Add TLS with Let’s Encrypt or use Caddy for simpler automatic HTTPS.

## Recommended Next Deployment Improvements

For a stronger internet deployment, the next steps should be:

1. Add authentication.
2. Store alerts and upload history in a database.
3. Move long EDF jobs to background workers.
4. Add model selection and model-specific upload validation in the UI.
5. Add CI/CD so deployments automatically run smoke tests.
