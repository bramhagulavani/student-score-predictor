import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Step 1: Load Saved Data from Phase 2 ────────────────────
print("Loading processed data from Phase 2...")
X_train = pickle.load(open(os.path.join(MODEL_DIR, "X_train.pkl"), "rb"))
X_test  = pickle.load(open(os.path.join(MODEL_DIR, "X_test.pkl"),  "rb"))
y_train = pickle.load(open(os.path.join(MODEL_DIR, "y_train.pkl"), "rb"))
y_test  = pickle.load(open(os.path.join(MODEL_DIR, "y_test.pkl"),  "rb"))
print("✅ Data loaded successfully")

# ── Step 2: Define All 3 Models ──────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(
                            n_estimators=100, random_state=42),
    "XGBoost"           : XGBRegressor(
                            n_estimators=100, random_state=42,
                            verbosity=0)
}

# ── Step 3: Train + Evaluate All Models ─────────────────────
results = {}

print("\n=== Training All 3 Models ===\n")
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "model"  : model,
        "y_pred" : y_pred,
        "MAE"    : round(mae,  2),
        "RMSE"   : round(rmse, 2),
        "R2"     : round(r2,   4)
    }

    print(f"  {name}")
    print(f"    MAE  : {mae:.2f}  (avg error in marks)")
    print(f"    RMSE : {rmse:.2f} (penalizes big errors)")
    print(f"    R2   : {r2:.4f}  (1.0 = perfect)\n")

# ── Step 4: Find Best Model ───────────────────────────────────
best_name = max(results, key=lambda x: results[x]["R2"])
best_model = results[best_name]["model"]
print(f"🏆 Best Model: {best_name} "
      f"(R2 = {results[best_name]['R2']})")

# ── Step 5: Chart 1 — Model Comparison Bar Chart ────────────
metrics_df = pd.DataFrame({
    "Model" : list(results.keys()),
    "MAE"   : [results[m]["MAE"]  for m in results],
    "RMSE"  : [results[m]["RMSE"] for m in results],
    "R2"    : [results[m]["R2"]   for m in results]
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sns.barplot(x="Model", y="MAE",  data=metrics_df,
            palette="Blues_d", ax=axes[0])
axes[0].set_title("MAE (lower = better)")
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(x="Model", y="RMSE", data=metrics_df,
            palette="Oranges_d", ax=axes[1])
axes[1].set_title("RMSE (lower = better)")
axes[1].tick_params(axis='x', rotation=15)

sns.barplot(x="Model", y="R2",   data=metrics_df,
            palette="Greens_d", ax=axes[2])
axes[2].set_title("R2 Score (higher = better)")
axes[2].tick_params(axis='x', rotation=15)

plt.suptitle("Model Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
plt.show()
print("✅ Chart 1 saved: outputs/model_comparison.png")

# ── Step 6: Chart 2 — Actual vs Predicted ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (name, res) in enumerate(results.items()):
    axes[i].scatter(y_test, res["y_pred"],
                    alpha=0.5, color='steelblue', s=20)
    axes[i].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2)
    axes[i].set_title(f"{name}\nR2={res['R2']}")
    axes[i].set_xlabel("Actual Score")
    axes[i].set_ylabel("Predicted Score")

plt.suptitle("Actual vs Predicted Scores",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.show()
print("✅ Chart 2 saved: outputs/actual_vs_predicted.png")

# ── Step 7: Chart 3 — Feature Importance (Best Model) ───────
if best_name in ["Random Forest", "XGBoost"]:
    feature_cols = pickle.load(
        open(os.path.join(MODEL_DIR, "features.pkl"), "rb"))
    importance = best_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature"    : feature_cols,
        "Importance" : importance
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature",
                data=feat_df, palette="viridis")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.show()
    print("✅ Chart 3 saved: outputs/feature_importance.png")

# ── Step 8: Save Best Model ───────────────────────────────────
pickle.dump(best_model,
            open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb"))
pickle.dump(best_name,
            open(os.path.join(MODEL_DIR, "best_model_name.pkl"), "wb"))
print(f"\n✅ Best model saved: models/best_model.pkl")

# ── Summary ──────────────────────────────────────────────────
print("\n========== PHASE 3 SUMMARY ==========")
for name, res in results.items():
    print(f"  {name:20} | MAE: {res['MAE']:5} "
          f"| RMSE: {res['RMSE']:5} | R2: {res['R2']}")
print(f"\n  🏆 Best Model : {best_name}")
print(f"  🎯 Best R2    : {results[best_name]['R2']}")
print("======================================")