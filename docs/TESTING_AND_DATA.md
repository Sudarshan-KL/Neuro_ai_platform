# Testing And Data Guide

This guide explains which tests to run, which data to use, and what each dataset is good for in the current state of the repo.

## Current Model Reality

The active checkpoint in the repo is:

- `saved_models/cnn_seizure_detector_v1.pt`

Its current metadata is:

- expected channels: `1`
- expected window size: `173`
- dataset: `bonn_epilepsy_eeg`

This is the most important testing fact in the project right now.

## Which Files Should Be Used For Current Upload Testing

Use these EDFs for real upload testing in the current deployment:

- [background_A_Z001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_A_Z001.edf)
- [background_B_O001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/background_B_O001.edf)
- [seizure_E_S001.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/bonn-edf-samples/seizure_E_S001.edf)

These were generated from the bundled Bonn text dataset specifically so the current model can process them through `/api/v1/upload_edf`.

## Which Files Should Not Be Used With The Current Model

These CHB-MIT files exist locally:

- [chb01_03.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/chb01/chb01_03.edf)
- [chb01_04.edf](c:/Users/rajku/Downloads/neuro-ai-platform/neuro-ai-platform/data/chb01/chb01_04.edf)

They are useful for future work, but they do not match the currently deployed model because they are multi-channel EEG recordings while the current model expects one channel.

## Where The Data Comes From

### Bonn data

Bundled in this repo under:

- `Epilepsy-EEG/`

Used for:

- current local training
- current deployed checkpoint
- generated EDF smoke-test files

### CHB-MIT data

Downloaded under:

- `data/chb01/`

Used for:

- future multi-channel training
- future multi-channel EDF deployment testing
- evaluating the original intended dataset path

## Recommended Test Strategy

There are three levels of testing you should care about.

### 1. Unit and API tests

Run the test suite:

```powershell
pytest tests -v
```

These tests cover:

- schema validation
- API endpoint behavior
- model architecture expectations
- streaming logic
- feature extraction
- data loading helpers

Important note:

- Many of these tests use mocked services, so they confirm app behavior without requiring the real deployed checkpoint.

### 2. Deployment smoke tests

Use these after starting the app:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/v1/health
Invoke-WebRequest http://127.0.0.1:8000/
Invoke-WebRequest http://127.0.0.1:8000/docs
```

Then test the real model path by:

- using the UI prediction controls
- uploading `seizure_E_S001.edf`
- checking `GET /api/v1/alerts`

### 3. Real file inference tests

Recommended order:

1. Upload `background_A_Z001.edf`
2. Upload `background_B_O001.edf`
3. Upload `seizure_E_S001.edf`
4. Compare alert volume across the three

That gives you a simple operator-level sanity check:

- background files should produce fewer alerts than seizure files
- seizure-like file should produce strong alert activity

## Best Test Cases To Use Right Now

If the goal is to validate the current deployed application, use these test cases:

### Test case 1. Health endpoint

Expected:

- `status = ok`
- `model_loaded = true`
- `expected_channels = 1`
- `expected_window_size = 173`

### Test case 2. UI loads

Open:

- `http://127.0.0.1:8000/`

Expected:

- 3D command deck loads
- telemetry panels populate
- active model shape is visible

### Test case 3. Synthetic single-window inference

Use the UI controls in the Prediction Lab.

Expected:

- request succeeds
- status and confidence update
- alerts may appear when the generated signal is seizure-like

### Test case 4. Compatible EDF upload

Upload:

- `seizure_E_S001.edf`

Expected:

- upload succeeds
- response includes file metadata
- alert history is populated

### Test case 5. Incompatible EDF rejection

Upload:

- `chb01_03.edf`

Expected:

- request is rejected
- error clearly states that the uploaded EDF has `23` channels but the active model expects `1`

This is a good test because it confirms that the app now fails clearly instead of failing ambiguously.

## Recommended Additional Tests To Add

The current repo would benefit from these extra tests:

1. A test asserting that `/health` returns model metadata such as expected channels and dataset.
2. A test asserting that EDF upload uses the active model’s `window_size` and `sfreq`.
3. A test asserting that incompatible EDF uploads return a clear shape-mismatch message.
4. A smoke test for the frontend route `/`.
5. A smoke test for the generated Bonn EDF samples.

## Suggested Future Testing Matrix

Once a CHB-MIT checkpoint exists, split tests into two model profiles:

- Bonn single-channel profile
- CHB-MIT multi-channel profile

Then validate:

- `predict_window`
- `predict_batch`
- `upload_edf`
- UI compatibility messaging
- alert generation behavior

## Useful Commands

### Run all tests

```powershell
pytest tests -v
```

### Skip slow tests

```powershell
pytest tests -v -m "not slow"
```

### Check current model metadata

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/v1/health
```

### Check alerts

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/v1/alerts
```

### Generate compatible EDF samples again

```powershell
$env:DEBUG='false'
python scripts/create_bonn_edf_samples.py
```

## Summary

If the question is “which test files should we use right now?”, the answer is:

- Use the generated Bonn EDF samples for current deployment validation.
- Use the CHB-MIT EDFs only as negative tests or as preparation for the next multi-channel model milestone.
