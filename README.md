# 🏭 Manufacturing Output Prediction

This project predicts **Parts Per Hour (Manufacturing Output)** using machine parameters and environmental conditions.

It helps industries **optimize production efficiency** by analyzing how different factors influence output.

---

## 🚀 Features

* 📊 Predicts production output (Parts Per Hour)
* ⚙️ Takes machine & environmental inputs
* 🧠 Uses Machine Learning (Regression models)
* 📈 Interactive dashboard using Streamlit
* ⭐ Shows important factors affecting output
* 📊 Visualizes input data and performance

---

## 🧠 Problem Statement

Manufacturing efficiency depends on multiple factors such as:

* Temperature
* Pressure
* Cycle Time
* Material Properties
* Operator Experience

This project predicts **production output**, helping industries improve performance and decision-making.

---

## ⚙️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Matplotlib
* Streamlit

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

* Injection Temperature
* Injection Pressure
* Cycle Time
* Cooling Time
* Material Viscosity
* Ambient Temperature
* Machine Age
* Operator Experience
* Maintenance Hours

### 📊 Categorical Inputs

* Machine Type (A, B, C)
* Material Grade (Premium, Standard)
* Day of Week

---

## 📌 Output

* ✅ Predicted **Parts Per Hour**
* 🔥 Efficiency Level:

  * High Efficiency (> 40)
  * Medium Efficiency (25–40)
  * Low Efficiency (< 25)

---

## 📊 Visualizations

* Input parameter bar chart
* Feature importance chart
* Efficiency indicator

---

## 📂 Project Structure

Manufacturing_Project/
│
├── app.py                # Streamlit application
├── model.pkl             # Trained ML model
├── requirements.txt      # Dependencies
├── manufacturing_dataset_1000_samples.csv
└── README.md

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/Srinidhisn5/manufacturing-output-prediction.git
cd manufacturing-output-prediction

---

### 2️⃣ Create Virtual Environment (Optional)

python -m venv venv

Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

### 3️⃣ Install Dependencies

pip install -r requirements.txt

---

## ▶️ Run the Application

streamlit run app.py

👉 The app will open in your browser automatically.

---

## 🎯 Use Case

* Optimize machine parameters
* Improve production efficiency
* Assist industrial decision-making

---

## 📌 Future Improvements

* Real-time data integration
* Cloud deployment (Streamlit Cloud / Render)
* Advanced analytics dashboard

---

## ⭐ Conclusion

This project demonstrates how machine learning can be used to **predict and improve manufacturing output**, making industrial processes more efficient and data-driven.
