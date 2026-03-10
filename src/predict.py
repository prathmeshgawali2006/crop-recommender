# ============================================================
# predict.py — CLI Crop Recommendation Tool
# ============================================================

import joblib
import numpy as np
import os

# ── LOAD THE TRAINED MODEL ──────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'crop_model.pkl')

print("🔄 Loading model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded!\n")

# ── CROP DESCRIPTIONS (for explainability) ──────────────────
CROP_INFO = {
    'rice':        'Thrives in high humidity and heavy rainfall.',
    'maize':       'Prefers moderate temperature and low humidity.',
    'chickpea':    'Thrives in low humidity and moderate rainfall.',
    'kidneybeans': 'Needs moderate rainfall and neutral pH.',
    'pigeonpeas':  'Tolerates dry conditions and low rainfall.',
    'mothbeans':   'Suitable for arid, low rainfall areas.',
    'mungbean':    'Prefers warm temperature and moderate humidity.',
    'blackgram':   'Grows well in warm, humid conditions.',
    'lentil':      'Prefers cool, dry climate with low humidity.',
    'pomegranate': 'Thrives in hot, dry climate with low rainfall.',
    'banana':      'Needs high humidity and heavy rainfall.',
    'mango':       'Prefers hot climate with low to moderate rainfall.',
    'grapes':      'Grows best in warm, dry climate.',
    'watermelon':  'Needs warm temperature and moderate water.',
    'muskmelon':   'Prefers hot, dry climate.',
    'apple':       'Requires cold climate and moderate rainfall.',
    'orange':      'Grows well in warm climate with moderate rainfall.',
    'papaya':      'Needs tropical, humid conditions.',
    'coconut':     'Requires high humidity and heavy rainfall.',
    'cotton':      'Prefers hot, dry climate with black soil.',
    'jute':        'Grows best in warm, humid, rainy conditions.',
    'coffee':      'Requires high humidity and heavy rainfall.',
}

# ── INPUT RANGES (for validation) ───────────────────────────
FEATURES = [
    ('Nitrogen (N)',       0,   140),
    ('Phosphorus (P)',     5,   145),
    ('Potassium (K)',      5,   205),
    ('Temperature (°C)',   8,    44),
    ('Humidity (%)',      14,   100),
    ('pH',               3.5,   9.5),
    ('Rainfall (mm)',     20,   300),
]

# ── FUNCTION: GET USER INPUT ─────────────────────────────────
def get_input():
    print("=" * 52)
    print("        🌾 CROP RECOMMENDATION SYSTEM")
    print("=" * 52)
    print("  Enter your soil and climate details below.\n")

    values = []
    for name, low, high in FEATURES:
        while True:
            try:
                val = float(input(f"  {name} ({low}–{high}): "))
                if low <= val <= high:
                    values.append(val)
                    break
                else:
                    print(f"    ⚠  Please enter a value between {low} and {high}\n")
            except ValueError:
                print("    ⚠  Please enter a valid number\n")

    return np.array(values).reshape(1, -1)

# ── FUNCTION: PREDICT TOP 3 CROPS ───────────────────────────
def predict(input_data):
    probabilities = model.predict_proba(input_data)[0]
    top3_indices  = np.argsort(probabilities)[::-1][:3]
    classes       = model.classes_

    return [(classes[i], probabilities[i] * 100) for i in top3_indices]

# ── FUNCTION: DISPLAY RESULTS ────────────────────────────────
def display_results(results):
    print("\n" + "=" * 52)
    print("           🌱 RECOMMENDATION RESULTS")
    print("=" * 52)

    best_crop, best_conf = results[0]
    print(f"\n  ✅ BEST CROP  :  {best_crop.upper()}")
    print(f"  📊 Confidence :  {best_conf:.1f}%")
    print(f"  💡 Why        :  {CROP_INFO.get(best_crop, 'Good match for your conditions.')}")

    print(f"\n  📋 ALTERNATIVES:")
    for rank, (crop, conf) in enumerate(results[1:], 2):
        print(f"\n  {rank}. {crop.capitalize()} ({conf:.1f}%)")
        print(f"     {CROP_INFO.get(crop, 'Suitable for your conditions.')}")

    print("\n" + "=" * 52)

# ── MAIN LOOP ────────────────────────────────────────────────
def main():
    while True:
        input_data = get_input()
        results    = predict(input_data)
        display_results(results)

        print()
        again = input("  🔄 Try another prediction? (yes / no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n  👋 Happy farming!\n")
            break

if __name__ == "__main__":
    main()