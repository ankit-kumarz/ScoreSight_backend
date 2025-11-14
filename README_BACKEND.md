# ScoreSight Backend (PyTorch)

This is a minimal Django + DRF backend that exposes endpoints for inference using TorchScript models.

Quick start

1. Create a Python virtual environment and activate it.

   powershell:

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

2. Place your sklearn model artifacts and metadata into `artifacts/` (project root):

- `match_outcome_classifier.pkl` (joblib / sklearn classifier)
- `goal_diff_regressor.pkl` (joblib / sklearn regressor)
- `feature_scaler.pkl` (optional - joblib scaler)
- `model_meta.json` (JSON with keys: `teams`, `features`, `class_labels`)

3. Run migrations and start the server:

   ```powershell
   python manage.py migrate
   python manage.py runserver
   ```

API Endpoints

- GET `/api/health` — basic health check
- GET `/api/teams` — returns teams from `artifacts/model_meta.json`
- POST `/api/predict_v2` — single-match prediction (JSON body with match stats)
- POST `/api/simulate` — upload CSV (Date, HomeTeam, AwayTeam) to simulate a table

Notes

- CORS is enabled (all origins) to allow simple frontends to call the API.
- Prediction history (optional) is stored in SQLite via `PredictionHistory` model. If DB is unavailable, predictions still work.
