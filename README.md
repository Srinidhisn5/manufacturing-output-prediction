# 🏭 Manufacturing Output Prediction

This project predicts **Parts Per Hour (Production Output)** using machine parameters and environmental conditions.

It helps in **optimizing manufacturing efficiency** by analyzing how different factors affect production.

---

## 🚀 Features

- 📊 Predicts manufacturing output (Parts per Hour)
- ⚙️ Uses multiple machine parameters as input
- 🧠 Machine Learning model (Regression)
- 📈 Interactive Streamlit dashboard
- ⭐ Shows key influencing factors (Feature Importance)
- 📊 Visualizations for better understanding

---

## 🧠 Problem Statement

In manufacturing industries, production efficiency depends on several factors like:

- Machine temperature  
- Pressure  
- Cycle time  
- Material properties  
- Operator experience  

This project predicts **how many parts can be produced per hour**, helping industries improve performance.

---

## ⚙️ Tech Stack

- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## 📊 Machine Learning Workflow

1. Data preprocessing (handling missing values, encoding)
2. Feature engineering (Temperature-Pressure ratio)
3. Model training (Linear Regression, Decision Tree, Random Forest)
4. Model evaluation (R², MAE, RMSE)
5. Model selection (best performing model)
6. Model deployment using Streamlit

---

## 📈 Input Parameters

### 🔢 Numeric Inputs
- Injection Temperature  
- Injection Pressure  
- Cycle Time  
- Cooling Time  
- Material Viscosity  
- Ambient Temperature  
- Machine Age  
- Operator Experience  
- Maintenance Hours  

### 📊 Categorical Inputs
- Machine Type (A, B, C)  
- Material Grade (Premium, Standard)  
- Day of Week  

---

## 📌 Output

- ✅ Predicted **Parts Per Hour**
- 🔥 Efficiency Level:
  - High Efficiency (> 40)
  - Medium Efficiency (25–40)
  - Low Efficiency (< 25)

---

## 📊 Visualizations

- Input parameter bar chart  
- Feature importance chart  
- Efficiency indicator  

---

## ▶️ How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Srinidhisn5/manufacturing-output-prediction.git
cd manufacturing-output-prediction
