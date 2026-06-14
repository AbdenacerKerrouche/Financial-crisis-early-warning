# 🧠 Systemic Risk Prediction

AI-powered system for predicting financial crises **3–6 months in advance** using network analysis, machine learning, and macro-financial indicators.

---

## 📌 Overview

This project was developed for the **Nexus Systemic Risk Prediction Challenge (2026)**.

Financial systems are highly interconnected. A failure in one institution can trigger cascading effects across the entire network. Traditional risk models fail to capture these dependencies.

This project builds an **early warning system** that:

* Detects systemic risk signals
* Predicts financial crises
* Estimates crisis severity
* Provides interpretable insights for decision-makers

---

## 🎯 Objectives

* Predict **crisis occurrence (binary classification)**
* Estimate **crisis severity (regression)**
* Incorporate **network effects** and macroeconomic indicators
* Deliver an **interactive dashboard** for real-time monitoring

---

## 🧱 Project Structure

```
├── code/
│   ├── Preprocessing & modeling.ipynb
│   ├── transactions EDA.ipynb
│
├── data/
│   ├── raw/
│   ├── train_df.csv
│   └── test_df.csv
│
├── models/
│   ├── best_clf_xgboost.pkl
│   └── best_reg_rf.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Sources

* Institution financial metrics
* Interbank exposure networks
* Market & macroeconomic indicators
* Transaction-level data
* Crisis labels

---

### 2. Feature Engineering

* Aggregation of institution-level metrics
* Network-derived features (exposures, connectivity)
* Temporal alignment of macro indicators
* Handling categorical + numerical pipelines

---

### 3. Models

#### 🔹 Classification Model

* **XGBoost**
* Predicts probability of crisis (`is_crisis`)

#### 🔹 Regression Model

* **Random Forest**
* Predicts crisis severity score

---

### 4. Interpretability

* SHAP used to:

  * Identify key risk drivers
  * Explain model decisions
  * Provide actionable insights

---

## 🚀 Deployment

The project includes a **Streamlit web application** for interactive use.

### ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🖥️ Application Features

* Input financial and macroeconomic indicators
* Predict:

  * Crisis probability
  * Crisis severity
* Visual interpretation of predictions
* Real-time inference using trained pipelines

---

## 🧪 Tech Stack

* **Python**
* **pandas, numpy**
* **scikit-learn**
* **XGBoost**
* **Random Forest**
* **SHAP**
* **Streamlit**

---

## 📊 Results

* Accurate early detection of crisis periods
* Robust performance across multiple indicators
* Interpretable outputs suitable for regulatory insights

---

## ⚠️ Limitations

* Simplified representation of real financial networks
* No full dynamic contagion simulation (future work)
* Model performance depends on feature engineering quality

---

## 🔮 Future Improvements

* Graph Neural Networks (GNNs) for network modeling
* Contagion simulation engine
* Real-time data integration
* API deployment (FastAPI)

---

## 📜 License

This project is for educational and research purposes.

---

## 👤 Author

Developed as part of the **Nexus Hackathon 2026**
Focus: **Systemic Risk • Machine Learning • Financial Networks**

---
