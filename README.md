# 🎓 Student Score Predictor

> An end-to-end Machine Learning project that predicts a student's **Math Score**
> based on personal and academic factors — comparing 3 regression models and
> serving live predictions through an interactive Streamlit web app.

**Built by Bramha Gulavani | VIT Pune | AI & ML Engineering | 2nd Year**

---

## 🖥️ Demo

![App Screenshot](outputs/app_screenshot.png)

> Select student details using dropdowns and sliders → AI predicts the Math Score instantly with grade

---

## 📊 Model Performance Comparison

| Model | R2 Score | MAE | RMSE | Winner |
|-------|----------|-----|------|--------|
| **Linear Regression** | **0.8838** | **4.13** | **5.32** | 🏆 |
| Random Forest | 0.8491 | 4.69 | 6.06 | |
| XGBoost | 0.8249 | 5.11 | 6.53 | |

> **Key Insight:** Linear Regression outperformed more complex models because the relationship
> between reading/writing scores and math scores is nearly linear — proving that
> simpler models can win when the data pattern is straightforward.

---

## 📌 Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Dataset loading + Exploratory Data Analysis | ✅ Done |
| Phase 2 | Feature Engineering + Label Encoding + Scaling | ✅ Done |
| Phase 3 | Train 3 Models + Compare + Evaluate | ✅ Done |
| Phase 4 | Streamlit Web App + Deployment | ✅ Done |

---

## 🛠️ Tech Stack

- **Language** — Python
- **ML Models** — Linear Regression, Random Forest, XGBoost
- **Libraries** — scikit-learn, pandas, numpy, XGBoost
- **Web App** — Streamlit
- **Visualization** — matplotlib, seaborn
- **Serialization** — pickle

---

## 📂 Folder Structure

```
student-score-predictor/
├── data/
│   └── raw/                          ← place StudentsPerformance.csv here
├── notebooks/                        ← EDA notebooks
├── src/
│   ├── phase1_eda.py                 ← EDA + 5 charts
│   ├── phase2_preprocessing.py       ← Feature engineering + encoding + scaling
│   ├── phase3_model.py               ← Train 3 models + compare
│   └── phase4_app.py                 ← Streamlit web app
├── models/                           ← saved best_model.pkl + scaler.pkl
├── outputs/                          ← saved charts
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/BramhaGulavani/student-score-predictor.git
cd student-score-predictor
```

**2. Create and activate virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run this once first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
from Kaggle and place it at:
```
data/raw/StudentsPerformance.csv
```

**5. Run each phase in order**
```bash
python src/phase1_eda.py
python src/phase2_preprocessing.py
python src/phase3_model.py
streamlit run src/phase4_app.py
```

---

## 🧠 How It Works

```
User selects student details (gender, lunch, test prep etc.)
                    ↓
Categorical values encoded to numbers (Label Encoding)
                    ↓
All values scaled to same range (Standard Scaler)
                    ↓
Linear Regression model predicts Math Score
                    ↓
Score + Grade + Progress bar displayed live
```

---

## 📈 Output Charts

| Chart | Description |
|-------|-------------|
| `score_distributions.png` | Math, Reading, Writing score distributions |
| `math_by_gender.png` | Math score comparison by gender |
| `math_by_testprep.png` | Impact of test preparation on scores |
| `correlation_heatmap.png` | Correlation between all 3 scores |
| `math_by_parentedu.png` | Math score by parental education level |
| `model_comparison.png` | MAE, RMSE, R2 comparison of 3 models |
| `actual_vs_predicted.png` | Actual vs predicted score scatter plots |
| `feature_correlation.png` | Feature importance with target variable |

---

## 💡 Key ML Concepts Learned

- **Regression vs Classification** — predicting a number vs predicting a category
- **Feature Engineering** — creating `average score`, `total score`, `pass/fail` columns
- **Label Encoding** — converting categorical text columns to numbers
- **Standard Scaling** — bringing all features to the same numerical range
- **Model Comparison** — training multiple models and picking the best one using R2, MAE, RMSE
- **Pickle** — saving and loading trained models for reuse

---

## ⚠️ Notes

- This project is trained on 1,000 student records — predictions are approximate
- Model average error is ±4.13 marks out of 100
- For best results, reading and writing scores are the strongest predictors
- This project is for educational and portfolio purposes

---

## 📄 License

Free to use for learning and portfolio purposes.

---

## 🙋 About Me

**Bramha Gulavani**
2nd Year AI & ML Engineering Student at VIT Pune
Building real ML projects from scratch — one phase at a time.

[![GitHub](https://img.shields.io/badge/GitHub-BramhaGulavani-black?style=flat&logo=github)](https://github.com/BramhaGulavani)

---

## 🗂️ My Other Projects

| Project | Description | Tech |
|---------|-------------|------|
| [🧠 Fake News Detector](https://github.com/BramhaGulavani/fake-news-detector) | Detects fake news using NLP + Logistic Regression | NLP, TF-IDF, Streamlit |
| [🎓 Student Score Predictor](https://github.com/BramhaGulavani/student-score-predictor) | Predicts student math scores using Regression | Linear Regression, XGBoost, Streamlit |