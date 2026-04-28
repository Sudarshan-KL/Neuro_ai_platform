# Multi-Disorder Workflow

This document covers the newly added Parkinson's, Alzheimer's, and neuro image tracks.

## 1) Train/prepare models

Run:

`python scripts/train_disorder_models.py`

This produces:

- `saved_models/alzheimers_image_classifier.pkl`
- `saved_models/parkinsons_tabular_classifier.pkl`
- `saved_models/neuro_image_classifier.pkl`

## 2) Run API

Run the FastAPI service as usual (for example `python -m app.main`).

## 3) Endpoints

- `POST /api/v1/predict/alzheimers` with `multipart/form-data` image file.
- `POST /api/v1/predict/parkinsons` with JSON containing any/all expected feature columns.
- `POST /api/v1/predict/neuro` with `multipart/form-data` image file.
- `GET /api/v1/model_info` for artifact and class metadata.
- `GET /api/v1/dataset_info` for dataset counts and schema info.
- `GET /api/v1/results` for recent in-memory prediction history.

## 4) Frontend

Open `/` and use the added disorder sections:

- Alzheimer's MRI upload and prediction panel
- Parkinson's feature JSON prediction panel
- Neuro image upload and prediction panel
- Objectives dashboard (model coverage + recent results)
