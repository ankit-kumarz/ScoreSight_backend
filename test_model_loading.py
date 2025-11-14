"""Quick test to verify model loading"""
import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from api.inference import get_inferencer

print("=" * 60)
print("🧪 Testing Model Loading")
print("=" * 60)

inf = get_inferencer()

print(f"\n📂 Base path: {inf.base}")
print(f"📊 Features count: {len(inf.features())}")
print(f"👥 Teams count: {len(inf.teams())}")
print(f"🏷️  Class labels: {inf.class_labels()}")

print(f"\n🤖 Classifier loaded: {'✅ YES' if inf.classifier is not None else '❌ NO'}")
print(f"📈 Regressor loaded: {'✅ YES' if inf.regressor is not None else '❌ NO'}")
print(f"⚖️  Scaler loaded: {'✅ YES' if inf.scaler is not None else '⚠️  NO (optional)'}")

# Test prediction
print("\n" + "=" * 60)
print("🧪 Testing Prediction with Sample Data")
print("=" * 60)

test_payload = {
    'HomeTeam': 'Chelsea',
    'AwayTeam': 'Brighton',
    'HTHG': 1,
    'HTAG': 0,
    'HS': 12,
    'AS': 8,
    'HST': 6,
    'AST': 3,
    'HC': 5,
    'AC': 3,
    'HF': 8,
    'AF': 10,
    'HY': 2,
    'AY': 1,
    'HR': 0,
    'AR': 0
}

print(f"\n📥 Input: {test_payload['HomeTeam']} vs {test_payload['AwayTeam']}")
print(f"   Half-time: {test_payload['HTHG']}-{test_payload['HTAG']}")
print(f"   Shots: {test_payload['HS']}-{test_payload['AS']}")
print(f"   Shots on target: {test_payload['HST']}-{test_payload['AST']}")

result = inf.predict_single(test_payload)

print(f"\n📤 Prediction Result:")
print(f"   Outcome: {result['outcome']}")
print(f"   Probabilities: {result['probabilities']}")
print(f"   Goal Difference: {result['goal_diff']}")
print(f"   Suggested Score: {result['suggested_score']}")

print("\n" + "=" * 60)
if inf.classifier is not None and inf.regressor is not None:
    print("✅ SUCCESS: Models are loaded and working!")
else:
    print("❌ WARNING: Using fallback demo logic (models not loaded)")
print("=" * 60)
