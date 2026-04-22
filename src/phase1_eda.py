import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "StudentsPerformance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Load Dataset ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ── Basic Info ───────────────────────────────────────────────
print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Shape (Rows x Columns) ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Basic Statistics ===")
print(df.describe())

# ── Chart 1: Score Distributions ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df['math score'], bins=20, kde=True,
             color='steelblue', ax=axes[0])
axes[0].set_title("Math Score Distribution")

sns.histplot(df['reading score'], bins=20, kde=True,
             color='seagreen', ax=axes[1])
axes[1].set_title("Reading Score Distribution")

sns.histplot(df['writing score'], bins=20, kde=True,
             color='coral', ax=axes[2])
axes[2].set_title("Writing Score Distribution")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_distributions.png"))
plt.show()
print("✅ Chart 1 saved: outputs/score_distributions.png")

# ── Chart 2: Math Score by Gender ────────────────────────────
plt.figure(figsize=(7, 4))
sns.boxplot(x='gender', y='math score', data=df, palette='Set2')
plt.title("Math Score by Gender")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "math_by_gender.png"))
plt.show()
print("✅ Chart 2 saved: outputs/math_by_gender.png")

# ── Chart 3: Math Score by Test Preparation ──────────────────
plt.figure(figsize=(7, 4))
sns.boxplot(x='test preparation course', y='math score',
            data=df, palette='Set3')
plt.title("Math Score by Test Preparation Course")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "math_by_testprep.png"))
plt.show()
print("✅ Chart 3 saved: outputs/math_by_testprep.png")

# ── Chart 4: Correlation Heatmap ─────────────────────────────
plt.figure(figsize=(6, 4))
corr = df[['math score', 'reading score', 'writing score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Scores")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.show()
print("✅ Chart 4 saved: outputs/correlation_heatmap.png")

# ── Chart 5: Math Score by Parental Education ────────────────
plt.figure(figsize=(10, 4))
order = df.groupby('parental level of education')['math score']\
          .mean().sort_values(ascending=False).index
sns.boxplot(x='parental level of education', y='math score',
            data=df, order=order, palette='Blues')
plt.xticks(rotation=20)
plt.title("Math Score by Parental Education Level")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "math_by_parentedu.png"))
plt.show()
print("✅ Chart 5 saved: outputs/math_by_parentedu.png")

# ── Summary ──────────────────────────────────────────────────
print("\n========== PHASE 1 SUMMARY ==========")
print(f"Total students       : {len(df)}")
print(f"Total columns        : {len(df.columns)}")
print(f"Missing values       : {df.isnull().sum().sum()}")
print(f"Avg math score       : {df['math score'].mean():.2f}")
print(f"Avg reading score    : {df['reading score'].mean():.2f}")
print(f"Avg writing score    : {df['writing score'].mean():.2f}")
print(f"Highest math score   : {df['math score'].max()}")
print(f"Lowest math score    : {df['math score'].min()}")
print("======================================")