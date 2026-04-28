# Test Assets For Demo And QA

This folder contains lightweight sample files for quickly showcasing the platform.

## Included assets

- `eeg_edf/seizure_E_S001.edf`
- `eeg_edf/background_A_Z001.edf`
- `alzheimers_images/alz_mild_00001.jpg`
- `alzheimers_images/alz_moderate_00001.jpg`
- `neuro_images/tumor_yes_Y1.jpg`
- `neuro_images/tumor_no_1.jpeg`
- `parkinsons/parkinsons_sample_25.csv`

## Suggested showcase flow

1. EEG seizure test:
   - Use `POST /api/v1/upload_edf` with a file from `eeg_edf/`.
2. Alzheimer's image prediction:
   - Use `POST /api/v1/predict/alzheimers` with a file from `alzheimers_images/`.
3. Neuro image prediction:
   - Use `POST /api/v1/predict/neuro` with a file from `neuro_images/`.
4. Parkinson's prediction:
   - Take one row from `parkinsons/parkinsons_sample_25.csv`
   - Send row values as JSON to `POST /api/v1/predict/parkinsons`.

These are copied from local `data/` so demos do not require browsing full datasets.
