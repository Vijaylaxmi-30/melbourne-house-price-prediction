# Internship Project Report – Melbourne House Price Prediction

**Submitted by:** Vijay Laxmi  
**Under the Supervision of:** Dr. Ganapathy Sannasi  
**Department:** Computer Science  
**Institution:** National Institute of Technical Teachers' Training & Research, Bhopal  
**Date:** June 2025  

---

## 📜 Certificate
This is to certify that the work embodied in this report entitled **Artificial and Machine Learning**, being submitted by *Vijay Laxmi* for partial fulfillment of the requirement for the award of Certificate during the short-term internship program, is a record of original work carried out under my supervision in the Department of Computer Science, NITTTR Bhopal.

---

## 🙋 Candidate’s Declaration
I, *Vijay Laxmi*, hereby declare that the report entitled **AI/ML** is my original work carried out during internship, and has not been submitted in part or in full to any other university or institution for the award of any certificate.

---

## 🙏 Acknowledgement
I express my gratitude to:
- **Dr. Ganapathy Sannasi** for guidance and support.
- All faculty members of the Department of Computer Science.
- My family and friends for their motivation.

---

## 📌 Executive Summary
This project focuses on predicting house prices in Melbourne using various Machine Learning models. The main steps involved were:
1. Data collection & cleaning.
2. Exploratory Data Analysis (EDA).
3. Feature engineering.
4. Model building and evaluation.

Models used:
- **Linear Regression**
- **Logistic Regression**
- **Random Forest**
- **LSTM (Long Short-Term Memory)**

---

## 🎯 Objectives
- Clean and preprocess the dataset.
- Explore and visualize influencing factors.
- Build and evaluate multiple ML models.
- Compare model performance.

---

## 🛠 Methodology
### 1. Data Preprocessing
- Dropped irrelevant columns.
- Handled missing values.
- Encoded categorical variables.
- Scaled numerical features.

---

### 2. Visualization
- Histograms for price distribution.
- Scatter plots for price vs. distance.
- Heatmaps for correlation analysis.
- Pie charts for property type distribution.

 --- 

### 3. Models Implemented
🔹 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train, y_train)

🔹 Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

🔹 LSTM
``python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

---
📊 Results & Discussion

| Model               | MAE       | RMSE        | R² Score | Accuracy | Precision | Recall | F1 Score |
|---------------------|-----------|-------------|----------|----------|-----------|--------|----------|
| Random Forest       | 997.24    | 16,475.47   | 0.9992   | 0.99     | 0.99      | 0.99   | 0.99     |
| LSTM                | 56,515.15 | 103,786.92  | 0.9686   | 0.95     | 0.94      | 0.95   | 0.94     |
| Logistic Regression | -         | -           | -        | 0.92     | 0.91      | 0.92   | 0.91     |


----

✅ Conclusion
Random Forest achieved the highest regression accuracy.
LSTM captured temporal patterns effectively.
Logistic Regression worked well for classification.

Future improvements: real-time data integration, economic indicators, and deployment as a web/mobile app.

📚 References
Housing Price Analysis Using Linear Regression and Logistic Regression
Dataset: Melbourne Housing Snapshot
