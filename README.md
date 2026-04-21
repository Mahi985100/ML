# 🌾 AgriPredict – Crop Yield Prediction System

## 📌 Overview

**AgriPredict** is a machine learning-based web application that predicts crop yield using soil nutrients and environmental data.
It helps farmers, students, and researchers make data-driven agricultural decisions.

---

## 🚀 Features

* 🌱 Predict crop yield based on field inputs
* 📊 Uses Random Forest ML model
* 🎨 Modern UI built with Streamlit
* ⚡ Fast and interactive predictions
* 📈 Visual insights and performance metrics
* 🧠 Supports multiple features:

  * Fertilizer
  * Temperature
  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * Rainfall
  * Humidity
  * Crop Season
  * Crop Name

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * plotly
  * joblib

---

## 📂 Project Structure

```
crop_yield_streamlit_app/
│
├── app.py                  # Main application (Home page)
├── pages/
│   ├── 1_About.py         # About page
│   ├── 2_Predict.py       # Prediction page
│
├── backend/
│   └── service.py         # ML processing and model logic
│
├── data/
│   └── crop_yield.csv     # Dataset
│
├── models/
│   └── model.pkl          # Trained ML model
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/AgriPredict.git
cd AgriPredict
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
python -m streamlit run app.py
```

---

## 📊 How It Works

1. User enters field data (soil + weather inputs)
2. Data is processed in backend
3. Machine learning model predicts yield
4. Result is displayed with visualization

---

## 📈 Model Details

* Algorithm: **Random Forest Regressor**
* Evaluation Metrics:

  * R² Score
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)

---

## 🎯 Use Cases

* Farmers for crop planning
* Students for ML projects
* Researchers for agricultural analysis

---

## 📸 Screenshots

(Add your screenshots here)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

**Mahi Nai**
BSc IT Student | ML Enthusiast

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
