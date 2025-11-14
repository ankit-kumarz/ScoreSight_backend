"""TorchScript model loader and inference helpers."""
"""SKLearn model loader and inference helpers."""
from pathlib import Path
import json
import numpy as np
import joblib
from typing import Optional


class SklearnInferencer:
    """Loader for scikit-learn models saved as .pkl in an artifacts/ folder.

    It looks for artifacts/model_meta.json and the following files:
      - artifacts/match_outcome_classifier.pkl
      - artifacts/goal_diff_regressor.pkl
      - artifacts/feature_scaler.pkl  (optional)
    """

    def __init__(self, base_path: Optional[str] = None):
        # attempt to locate artifacts directory in common places
        base = Path(base_path) if base_path else None
        if base is None:
            # project root (two levels up from this file)
            candidate = Path(__file__).resolve().parent.parent.parent / 'artifacts'
            if candidate.exists():
                base = candidate
            else:
                # fallback to sibling artifacts under backend
                candidate2 = Path(__file__).resolve().parent.parent / 'artifacts'
                base = candidate2
        self.base = Path(base)
        self.meta = {}
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self._load_meta()
        self._load_models()

    def _load_meta(self):
        meta_path = self.base / 'model_meta.json'
        if not meta_path.exists():
            # try legacy path under project root
            alt = Path(__file__).resolve().parent.parent.parent / 'artifacts' / 'model_meta.json'
            if alt.exists():
                meta_path = alt
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def _load_models(self):
        try:
            cls_path = self.base / 'match_outcome_classifier.pkl'
            reg_path = self.base / 'goal_diff_regressor.pkl'
            scaler_path = self.base / 'feature_scaler.pkl'

            if cls_path.exists():
                self.classifier = joblib.load(str(cls_path))
                print(f"✅ Loaded classifier from {cls_path}")
            else:
                print(f"❌ Classifier not found at {cls_path}")
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            self.classifier = None

        try:
            if reg_path.exists():
                self.regressor = joblib.load(str(reg_path))
                print(f"✅ Loaded regressor from {reg_path}")
            else:
                print(f"❌ Regressor not found at {reg_path}")
        except Exception as e:
            print(f"❌ Error loading regressor: {e}")
            self.regressor = None

        try:
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
                print(f"✅ Loaded scaler from {scaler_path}")
            else:
                print(f"⚠️ Scaler not found at {scaler_path} (optional)")
        except Exception as e:
            print(f"⚠️ Error loading scaler: {e}")
            self.scaler = None

    def teams(self):
        return self.meta.get('teams', [])

    def features(self):
        return self.meta.get('features', [])

    def class_labels(self):
        return self.meta.get('class_labels', ['H', 'D', 'A'])

    def _build_vector(self, payload: dict):
        """Build feature vector from match statistics.
        
        The trained model expects rolling average features (last 3 and last 5 matches).
        Since we don't have historical data in real-time prediction, we'll use
        current match stats as approximations for the rolling averages.
        """
        features = self.features()
        
        # Extract current match stats from payload
        # Half-time goals
        hthg = float(payload.get('HTHG', 0))
        htag = float(payload.get('HTAG', 0))
        
        # Shots
        hs = float(payload.get('HS', 0))
        as_shots = float(payload.get('AS', 0))
        hst = float(payload.get('HST', 0))
        ast = float(payload.get('AST', 0))
        
        # Fouls
        hf = float(payload.get('HF', 0))
        af = float(payload.get('AF', 0))
        
        # Corners
        hc = float(payload.get('HC', 0))
        ac = float(payload.get('AC', 0))
        
        # Cards
        hy = float(payload.get('HY', 0))
        ay = float(payload.get('AY', 0))
        hr = float(payload.get('HR', 0))
        ar = float(payload.get('AR', 0))
        
        # Build feature vector matching the trained model's expected features
        # The model expects: [H_goals_last3_mean, H_shots_last3_mean, ..., A_red_last5_mean]
        vec = []
        
        for f in features:
            # Map rolling average features to current match stats
            # Since we don't have historical data, use current stats as proxy
            
            # Home team last 3 match averages
            if 'H_goals_last3' in f:
                vec.append(hthg)  # Use half-time goals as proxy
            elif 'H_shots_last3' in f:
                vec.append(hs)
            elif 'H_sot_last3' in f:
                vec.append(hst)  # shots on target
            elif 'H_fouls_last3' in f:
                vec.append(hf)
            elif 'H_corners_last3' in f:
                vec.append(hc)
            elif 'H_yellow_last3' in f:
                vec.append(hy)
            elif 'H_red_last3' in f:
                vec.append(hr)
            
            # Away team last 3 match averages
            elif 'A_goals_last3' in f:
                vec.append(htag)
            elif 'A_shots_last3' in f:
                vec.append(as_shots)
            elif 'A_sot_last3' in f:
                vec.append(ast)
            elif 'A_fouls_last3' in f:
                vec.append(af)
            elif 'A_corners_last3' in f:
                vec.append(ac)
            elif 'A_yellow_last3' in f:
                vec.append(ay)
            elif 'A_red_last3' in f:
                vec.append(ar)
            
            # Home team last 5 match averages (use same values)
            elif 'H_goals_last5' in f:
                vec.append(hthg)
            elif 'H_shots_last5' in f:
                vec.append(hs)
            elif 'H_sot_last5' in f:
                vec.append(hst)
            elif 'H_fouls_last5' in f:
                vec.append(hf)
            elif 'H_corners_last5' in f:
                vec.append(hc)
            elif 'H_yellow_last5' in f:
                vec.append(hy)
            elif 'H_red_last5' in f:
                vec.append(hr)
            
            # Away team last 5 match averages (use same values)
            elif 'A_goals_last5' in f:
                vec.append(htag)
            elif 'A_shots_last5' in f:
                vec.append(as_shots)
            elif 'A_sot_last5' in f:
                vec.append(ast)
            elif 'A_fouls_last5' in f:
                vec.append(af)
            elif 'A_corners_last5' in f:
                vec.append(ac)
            elif 'A_yellow_last5' in f:
                vec.append(ay)
            elif 'A_red_last5' in f:
                vec.append(ar)
            
            else:
                # Fallback for any unknown features
                vec.append(0.0)
        
        return np.array(vec, dtype=np.float32).reshape(1, -1)

    def predict_single(self, payload: dict):
        """Predict using sklearn models and metadata.

        Returns dict with same shape as before.
        """
        res = {'outcome': None, 'probabilities': [], 'goal_diff': None, 'suggested_score': {'home': 0, 'away': 0}}

        X = self._build_vector(payload)
        # apply scaler if present
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                pass

        labels = self.class_labels()

        if self.classifier is not None:
            try:
                probs = self.classifier.predict_proba(X)[0]
            except Exception:
                # some classifiers may not have predict_proba
                preds = self.classifier.predict(X)
                probs = np.zeros(len(labels), dtype=float)
                for i, lab in enumerate(labels):
                    probs[i] = float(np.mean(preds == lab))
            res['probabilities'] = [{'label': lab, 'prob': float(round(float(p), 4))} for lab, p in zip(labels, probs)]
            res['outcome'] = labels[int(np.argmax(probs))]

        if self.regressor is not None:
            try:
                gd = float(self.regressor.predict(X)[0])
            except Exception:
                gd = 0.0
            res['goal_diff'] = float(gd)
            diff = int(round(gd))
            if diff == 0:
                res['suggested_score'] = {'home': 1, 'away': 1}
            elif diff > 0:
                res['suggested_score'] = {'home': max(1, 1 + diff), 'away': 1}
            else:
                res['suggested_score'] = {'home': 1, 'away': max(1, 1 + abs(diff))}

        # if no models loaded, still return a simple deterministic demo (keep UI usable)
        if self.classifier is None and self.regressor is None:
            teams = self.teams()
            team2idx = {t: i for i, t in enumerate(teams)}
            hidx = team2idx.get(payload.get('HomeTeam'), 0)
            aidx = team2idx.get(payload.get('AwayTeam'), 0)
            ht = float(payload.get('HTHG') or 0)
            at = float(payload.get('HTAG') or 0)
            base = (hidx - aidx) * 0.05 + (ht - at) * 0.25
            gd = max(min(base, 3.0), -3.0)
            res['goal_diff'] = float(round(gd, 2))
            logits = np.array([gd, 0.0, -gd], dtype=np.float32)
            exp = np.exp(logits - np.max(logits))
            probs = (exp / exp.sum()).tolist()
            res['probabilities'] = [{'label': lab, 'prob': float(round(float(p), 4))} for lab, p in zip(labels, probs)]
            res['outcome'] = labels[int(np.argmax(probs))]
            diff = int(round(gd))
            if diff == 0:
                res['suggested_score'] = {'home': 1, 'away': 1}
            elif diff > 0:
                res['suggested_score'] = {'home': max(1, 1 + diff), 'away': 1}
            else:
                res['suggested_score'] = {'home': 1, 'away': max(1, 1 + abs(diff))}

        return res


# module-level instance
_inferencer = None


def get_inferencer():
    global _inferencer
    if _inferencer is None:
        _inferencer = SklearnInferencer()
    return _inferencer

