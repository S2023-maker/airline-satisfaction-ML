# ✈️ Airline Passenger Satisfaction Predictor

A machine learning web app that predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** based on their flight experience.

**Image** <img width="1722" height="850" alt="image" src="https://github.com/user-attachments/assets/f0c2f4b5-e8c1-4d80-b0c6-be29bd7527c0" />


🔗 **Live App:** https://airline-satisfaction-ml-wcvnwp83tnfjfygxpqtnyx.streamlit.app/

---

## 📌 Project Overview

This project was built in 3 sprints:

- **Sprint 1** — Data Understanding & Preprocessing (EDA, cleaning, encoding)
- **Sprint 2** — Model Building & Evaluation (5 models compared)
- **Sprint 3** — Hyperparameter Tuning & Final Model Deployment

---

## 🏆 Final Model

| Model | Test Accuracy | F1 Score |
|---|---|---|
| Gradient Boosting (Tuned) | 96.7% | 96.1% |

The final model is a tuned **Gradient Boosting Classifier** wrapped in a full sklearn Pipeline (preprocessing + model).

---

## 📊 Dataset

**Airline Passenger Satisfaction** dataset with 25+ features including:
- Passenger info (age, gender, customer type, travel class)
- Flight details (distance, departure/arrival delay)
- 14 service ratings (seat comfort, food, WiFi, entertainment, etc.)

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/S2023-maker/airline-satisfaction-ML.git
cd airline-satisfaction-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
airline-satisfaction-ML/
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── final_model_pipeline.pkl    # Trained model pipeline
├── label_encoder.pkl           # Target label encoder
├── feature_meta.pkl            # Feature metadata
└── notebooks/
    ├── sprint1_data_preprocessing.ipynb
    └── sprint2_model_building.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Scikit-learn** — ML models & pipelines
- **Pandas / NumPy** — Data processing
- **Streamlit** — Web app deployment
- **Matplotlib / Seaborn** — EDA visualizations

---

## 📈 Models Compared

| Model | Test Accuracy | F1 Score |
|---|---|---|
| Gradient Boosting (Tuned) | 96.7% | 96.1% |
| Random Forest (Tuned) | 96.5% | 95.9% |
| Logistic Regression | 87.2% | 85.1% |
| Decision Tree | 94.1% | 93.5% |
| Naïve Bayes | 85.3% | 83.2% |

---

## ✍️ Author

**Shreya R Joshi**  
[GitHub](https://github.com/S2023-maker)
