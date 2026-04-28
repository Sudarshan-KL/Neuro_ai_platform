# Neuro-AI Seizure Detection Platform

Neuro-AI is an EEG seizure-detection platform built around a FastAPI backend, a production-style frontend command deck, and a CNN-based inference pipeline.

At this point, the repo contains two important tracks:

- The original CHB-MIT-oriented multi-channel pipeline and API structure.
- A working Bonn single-channel deployment path that is currently the active runnable model in this repo.

This README explains what we built, how the project evolved, what is running now, and where to go next. For day-to-day usage and deployment steps, use the supporting guides:

- [Run And Deploy Guide](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/docs/RUN_AND_DEPLOY.md)
- [Testing And Data Guide](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/docs/TESTING_AND_DATA.md)
- [Multi-Disorder Workflow](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/docs/MULTI_DISORDER_WORKFLOW.md)

## Step-By-Step: Run The Project Locally

Follow these steps from the project root (`neuro-ai-platform/neuro-ai-platform`):

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. (Optional but recommended) prepare multi-disorder model artifacts.

```powershell
python scripts/train_disorder_models.py
```

4. Set DEBUG to a safe value for local startup.

```powershell
$env:DEBUG='false'
```

5. Start the FastAPI app.

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

6. Open the app in your browser.

- UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`

7. Use ready-made demo files to test quickly.

- `tests/test_assets/eeg_edf/` for EEG EDF uploads
- `tests/test_assets/alzheimers_images/` for Alzheimer's image prediction
- `tests/test_assets/neuro_images/` for neuro image prediction
- `tests/test_assets/parkinsons/parkinsons_sample_25.csv` for Parkinson's payload rows

8. Stop the app when done with `Ctrl + C` in the terminal.

## Multi-Disorder Expansion (Current Sprint)

The platform now includes scaffolding for three disorder tracks in addition to the EEG seizure flow:

- Alzheimer's MRI image classification
- Parkinson's tabular voice-feature classification
- Neuro image classification (brain tumor / other)

### New data loaders and preprocessing

- `ml/data_loader/parkinsons_loader.py` for CSV loading and feature/target split.
- `ml/data_loader/neuro_loader.py` for generic image-folder class scanning.
- `ml/preprocessing/multimodal.py` for image normalization/vectorization and tabular preprocessing.

### New training/inference artifacts

- `app/services/disorder_models.py` manages lightweight model training/loading/saving for all 3 disorder tracks.
- `scripts/train_disorder_models.py` bootstraps and persists the three model artifacts in `saved_models/`.

### New API endpoints

- `POST /api/v1/predict/alzheimers` (image upload)
- `POST /api/v1/predict/parkinsons` (JSON feature payload)
- `POST /api/v1/predict/neuro` (image upload)
- `GET /api/v1/model_info`
- `GET /api/v1/dataset_info`
- `GET /api/v1/results`

### Frontend additions

The `/` UI now contains:

- Dedicated Alzheimer's upload/predict panel
- Dedicated Parkinson's JSON-payload prediction panel
- Dedicated neuro image upload/predict panel
- Objective coverage dashboard using model/dataset/results endpoints

## What We Built

The system has four main layers:

- Data loading and preprocessing for EEG recordings.
- Model training and model artifact management.
- A FastAPI inference and monitoring service.
- A separate, now integrated, frontend UI served from the backend.

The frontend is available from the same FastAPI service at `/`, so the app can be deployed as a single web service rather than as a split frontend/backend stack.

## Current Working State

The current deployed checkpoint in this repo is:

- `saved_models/cnn_seizure_detector_v1.pt`

That checkpoint is a Bonn Epilepsy-EEG model with:

- `1` input channel
- `173` sample window size
- dataset metadata `bonn_epilepsy_eeg`

This matters because:

- Bonn-compatible inputs work end to end.
- CHB-MIT-style `23`-channel EDF files do not work with the currently active model.

The API now exposes that expected input shape at `GET /api/v1/health`, and the UI surfaces the same information to reduce operator confusion.

## Project Journey

We effectively built the project in phases:

1. The base repo already had a strong architecture for CHB-MIT ingestion, training, inference, and testing.
2. We brought the backend up, trained a practical model locally using the bundled Bonn dataset, and saved a runnable checkpoint.
3. We fixed multiple compatibility issues in training and model loading so the trained checkpoint could actually be served.
4. We created a cinematic standalone frontend UI and then promoted it into a production-style same-origin frontend served by FastAPI.
5. We added real UI actions for telemetry, prediction, EDF upload, and alert monitoring.
6. We fixed the EDF upload flow so it uses the active model’s real sampling and window configuration instead of stale defaults.
7. We generated compatible single-channel EDF test files from the bundled Bonn dataset so the upload flow can be validated locally.

## Key Fixes Made

The repo needed several practical fixes to become runnable end to end:

- `scripts/train_bonn.py`
  The training loop was mismatched to the CNN output shape. It now trains correctly as a binary sigmoid model.
- `app/models/model_registry.py`
  The checkpoint loader now works with the installed PyTorch behavior and preserves checkpoint metadata.
- `app/main.py`
  The inference service now adopts the active model’s real `window_size` instead of assuming CHB-MIT defaults.
- `app/api/routes.py`
  EDF upload now respects active model metadata such as `sfreq`, `window_size`, and `window_stride`.
- `app/services/inference.py`
  Input shape validation now fails clearly when payloads do not match the active model.
- `frontend/`
  The UI was upgraded from a static visual concept into a live operator console with deployment-safe same-origin API calls.

## Architecture Overview

The current architecture looks like this:

1. EEG data enters through raw arrays or EDF uploads.
2. Data is loaded and segmented into windows.
3. The active CNN model scores seizure probability.
4. The inference service converts probability into `ALERT` or `CLEAR`.
5. Alerts are exposed through API endpoints and rendered in the command deck UI.

The backend exposes:

- `GET /`
  Frontend UI
- `GET /docs`
  Swagger UI
- `GET /api/v1/health`
  Service + model metadata
- `GET /api/v1/models`
  Available model versions
- `GET /api/v1/alerts`
  Alert history
- `POST /api/v1/predict_window`
  Single-window inference
- `POST /api/v1/predict_batch`
  Batch inference
- `POST /api/v1/stream_detect`
  Streaming chunk ingestion
- `POST /api/v1/upload_edf`
  EDF processing

## Frontend Direction

The UI intentionally does not look like a generic admin dashboard. It uses:

- 3D orbital motion graphics
- glassmorphism panels
- animated EEG waveform displays
- live telemetry panels
- operator-focused prediction and upload actions

That visual direction is now preserved in the production path, not just in a mockup.

## What Works Right Now

Right now you can:

- Start the backend and open the UI at `http://127.0.0.1:8000/`
- Inspect API docs at `http://127.0.0.1:8000/docs`
- Run live synthetic Bonn-shaped inference from the UI
- Upload compatible single-channel EDF files
- Inspect alerts in the UI and through the API

Quick demo assets are also prepared under `tests/test_assets/` for fast endpoint showcases.

Compatible EDF samples are available at:

- [background_A_Z001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_A_Z001.edf)
- [background_B_O001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_B_O001.edf)
- [seizure_E_S001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/seizure_E_S001.edf)

## Current Limitation

The biggest current limitation is model/data alignment.

The repo contains CHB-MIT sample EDFs under `data/chb01/`, but the active deployed model is Bonn-based and expects one channel. That means:

- Bonn sample EDFs are valid for current deployment testing.
- CHB-MIT EDFs are useful as future training/evaluation targets, but not for the current active checkpoint.

## How We Should Proceed Further

The most valuable next steps are:

1. Add multi-model support in production.
   The UI and backend should allow explicit selection between a Bonn single-channel model and a CHB-MIT multi-channel model.
2. Train and register a real multi-channel CHB-MIT checkpoint.
   That will make the uploaded `23`-channel EDFs first-class citizens.
3. Persist alerts and job history.
   Right now alert history is in memory. For production, it should live in a database.
4. Add asynchronous EDF job processing.
   Large EDF analysis should move to a background job queue rather than being fully synchronous.
5. Add authentication and environment-specific CORS.
   For real internet deployment, public access should not be anonymous.
6. Add CI validation for the active deployment mode.
   A smoke test should confirm that the active checkpoint, API shape, and upload flow all match.
7. Add richer model introspection.
   The UI should show expected channels, expected window size, dataset origin, and artifact metadata everywhere operator decisions matter.

## Recommended Near-Term Roadmap

If we want the next practical milestone, this is the order I’d recommend:

1. Introduce `v2` or `chbmit_v1` as a new checkpoint with true multi-channel support.
2. Add model selection to the UI.
3. Add a tiny persistence layer for uploads and alerts.
4. Add deployment settings for domain, TLS termination, and origin restrictions.
5. Add a CI job that boots the app and runs one inference smoke test with a compatible sample file.

## Repository Guides

Use these supporting docs depending on what you need:

- [Run And Deploy Guide](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/docs/RUN_AND_DEPLOY.md)
- [Testing And Data Guide](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/docs/TESTING_AND_DATA.md)
