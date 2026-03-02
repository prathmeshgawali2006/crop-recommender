# =============================================
# CROP RECOMMENDATION SYSTEM - MODEL TRAINING
# =============================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=" * 50)
print("   CROP RECOMMENDATION - MODEL TRAINING")
print("=" * 50)

# ----------------------
# STEP 1 - LOAD DATA
# ----------------------
print("\n📂 Loading dataset...")
df = pd.read_csv(r"C:\Users\asus\Desktop\crop-recommender\data\Crop_recommendation.csv")
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------
# STEP 2 - PREPARE DATA
# ----------------------
print("\n⚙️  Preparing data...")

# Separate features (X) and target (y)
X = df.drop("label", axis=1)   # everything except crop name
y = df["label"]                 # just the crop name

print(f"Features (X): {list(X.columns)}")
print(f"Target (y): crop label — {y.nunique()} unique crops")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42     # ensures same split every time you run
)

print(f"\n📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples:  {len(X_test)}")

# ----------------------
# STEP 3 - TRAIN MODEL
# ----------------------
print("\n🧠 Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ----------------------
# STEP 4 - EVALUATE
# ----------------------
print("\n📈 Evaluating model...")

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Detailed Report:")
print(classification_report(y_test, predictions))

# ----------------------
# STEP 5 - SAVE CHARTS
# ----------------------
print("\n💾 Saving evaluation charts...")

# Chart 1 — Feature Importance
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
feature_importance.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Feature Importance — Which factors matter most?", fontsize=13)
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/feature_importance.png")
plt.close()
print("✅ Feature importance chart saved!")

# Chart 2 — Confusion Matrix
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Confusion Matrix", fontsize=14)
plt.ylabel("Actual Crop")
plt.xlabel("Predicted Crop")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved!")

# ----------------------
# STEP 6 - SAVE MODEL
# ----------------------
print("\n💾 Saving trained model...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/crop_model.pkl")
print("✅ Model saved to model/crop_model.pkl")

print("\n" + "=" * 50)
print("   TRAINING COMPLETE! 🎉")
print("=" * 50)