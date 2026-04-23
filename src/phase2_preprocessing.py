import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "StudentsPerformance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Load Dataset ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ── Step 1: Feature Engineering ──────────────────────────────
print("\n=== Step 1: Feature Engineering ===")

# Create average score column
df['average score'] = (df['math score'] +
                       df['reading score'] +
                       df['writing score']) / 3

# Create pass/fail column (pass = average >= 40)
df['pass/fail'] = df['average score'].apply(
    lambda x: 'Pass' if x >= 40 else 'Fail'
)

# Create total score column
df['total score'] = (df['math score'] +
                     df['reading score'] +
                     df['writing score'])

print("New columns created:")
print(df[['math score', 'reading score',
          'writing score', 'average score',
          'total score', 'pass/fail']].head())

# ── Step 2: Encode Categorical Columns ───────────────────────
print("\n=== Step 2: Encoding Categorical Columns ===")

# List all categorical columns
cat_cols = ['gender', 'race/ethnicity',
            'parental level of education',
            'lunch', 'test preparation course']

print("Before encoding:")
print(df[cat_cols].head(3))

# Apply Label Encoding to each categorical column
le = LabelEncoder()
for col in cat_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])

print("\nAfter encoding:")
encoded_cols = [c + '_encoded' for c in cat_cols]
print(df[encoded_cols].head(3))

# ── Step 3: Define Features and Target ───────────────────────
print("\n=== Step 3: Features and Target ===")

# Features = everything we use to predict
# Target = what we want to predict (math score)
feature_cols = encoded_cols + ['reading score', 'writing score']
target_col   = 'math score'

X = df[feature_cols]
y = df[target_col]

print(f"Features used : {feature_cols}")
print(f"Target        : {target_col}")
print(f"X shape       : {X.shape}")
print(f"y shape       : {y.shape}")

# ── Step 4: Train Test Split ──────────────────────────────────
print("\n=== Step 4: Train Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Step 5: Feature Scaling ───────────────────────────────────
print("\n=== Step 5: Feature Scaling ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Scaling done ✅")
print(f"Sample scaled values: {X_train_scaled[0][:4]}")

# ── Chart: Feature Importance Preview ────────────────────────
plt.figure(figsize=(8, 4))
corr_with_target = df[feature_cols + [target_col]].corr()[target_col]\
                     .drop(target_col).sort_values(ascending=False)
sns.barplot(x=corr_with_target.values,
            y=corr_with_target.index, palette='coolwarm')
plt.title("Feature Correlation with Math Score")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"))
plt.show()
print("✅ Chart saved: outputs/feature_correlation.png")

# ── Step 6: Save Everything for Phase 3 ──────────────────────
pickle.dump(X_train_scaled,
            open(os.path.join(MODEL_DIR, "X_train.pkl"), "wb"))
pickle.dump(X_test_scaled,
            open(os.path.join(MODEL_DIR, "X_test.pkl"),  "wb"))
pickle.dump(y_train,
            open(os.path.join(MODEL_DIR, "y_train.pkl"), "wb"))
pickle.dump(y_test,
            open(os.path.join(MODEL_DIR, "y_test.pkl"),  "wb"))
pickle.dump(scaler,
            open(os.path.join(MODEL_DIR, "scaler.pkl"),  "wb"))
pickle.dump(feature_cols,
            open(os.path.join(MODEL_DIR, "features.pkl"),"wb"))

# Save cleaned df for Phase 4 app
df.to_csv(os.path.join(BASE_DIR, "data", "raw",
                        "students_cleaned.csv"), index=False)

print("\n✅ All files saved to models/ folder")
print("\n========== PHASE 2 SUMMARY ==========")
print(f"Total students       : {len(df)}")
print(f"Features used        : {len(feature_cols)}")
print(f"Training samples     : {len(X_train)}")
print(f"Testing samples      : {len(X_test)}")
print(f"Categorical columns  : {len(cat_cols)} encoded")
print(f"Target column        : {target_col}")
print("======================================")