
# 🧠 Extrovert vs Introvert Personality Prediction

Welcome to the **Extrovert/Introvert Personality Prediction** project – a behavioral machine learning analysis that classifies people as *Extroverts* or *Introverts* based on their lifestyle and social activity patterns.

> 📌 [Kaggle Competition – Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7)  
> 📁 [Project GitHub Repository](https://github.com/Eddythemachine/extrovert_introvert_prediction_kaggle_competition)

---

## 📌 Project Summary

Using real-world personality data, we performed extensive EDA, cleaned and transformed features, handled outliers, and trained multiple classification models. The pipeline includes handling missing values, personality-aware feature engineering, and ensemble learning with models like LightGBM, XGBoost, and Stacking Classifier.

---

## 📁 Project Structure

```
EXTROVERT_INTROVERT_PREDICTION_KAGGLE_COMPETITION/
│
├── data/
│   ├── cleaned/
│   │   ├── test_processed.csv
│   │   ├── train_processed.csv
│   │   └── val_processed.csv
│   │
│   ├── raw/
│   │   ├── test.csv
│   │   └── train.csv
│   │
│   └── submission/
│       ├── submission_full.csv
│       └── sample_submission.csv
│
├── model/
│   ├── data_cleaning/
│   │   └── preprocessing_pipeline.pkl
│   │
│   └── modelling/
│       └── final_full_model.pkl
│
├── notebook/
│   ├── data_cleaning.ipynb
│   ├── eda.ipynb
│   ├── model.ipynb
│   └── feature_parameters.csv
│
├── output/
│   ├── eda_report.txt
│   ├── feature_eng.txt
│   └── report.doc
│
├── report/
│   └── .gitignore
│
├── README.md
└── requirements.txt

```

---

## 🔍 Key Features Analyzed

- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`
- `Personality` (Target)

---

## 📈 Highlights from EDA

- Majority are **Extroverts** (≈74%)
- **Introverts** showed higher outlier rates in behavioral traits
- Features such as `Stage_fear` and `Drained_after_socializing` provide strong predictive signals
- Outlier and binning strategy applied to numerical traits (e.g., `Time_spent_Alone`)

📖 See full analysis in [`report.doc`](./report.doc)

---

## 🤖 Models Used

- Random Forest  
- Gradient Boosting  
- Stacking Classifier  
- Logistic Regression  
- KNeighborsClassifier  
- SVM  
- XGBoost  
- LightGBM  

✅ Evaluated using accuracy score and stratified training with `RandomizedSearchCV`

---

## ⚙️ Setup & Installation

### 🔧 Local Setup (VS Code or Terminal)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Eddythemachine/extrovert_introvert_prediction_kaggle_competition.git
   cd extrovert_introvert_prediction_kaggle_competition
   ```

2. **(Optional)** Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open notebooks using:**
   - Jupyter: `jupyter notebook`
   - VS Code: Open `.ipynb` files using Jupyter extension

---

### ☁️ Google Colab Setup

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Open Notebook → GitHub**
3. Paste this repo link:  
   `https://github.com/Eddythemachine/extrovert_introvert_prediction_kaggle_competition`
4. Select a notebook (e.g., `eda.ipynb`)
5. Run this cell at the top to install all requirements:
   ```python
   !pip install -r https://raw.githubusercontent.com/Eddythemachine/extrovert_introvert_prediction_kaggle_competition/main/requirements.txt
   ```

---

## 📦 requirements.txt Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

---

## 👤 Author

**Edison Onayifeke**  
📧 edisononayifeke@gmail.com  
🔗 [GitHub: Eddythemachine](https://github.com/Eddythemachine)  
🔗 [LinkedIn: oghenerunor](https://www.linkedin.com/in/oghenerunor/)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE)
