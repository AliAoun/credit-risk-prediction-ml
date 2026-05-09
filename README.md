# 💳 Credit Risk Prediction ML

A sophisticated machine learning application for predicting credit risk using XGBoost with model explainability powered by SHAP. Built with a modern Streamlit web interface for real-time predictions and intuitive feature analysis.

---

## 🎨 Application Screenshots

### Main Dashboard
<img width="1920" height="1080" alt="Credit Risk Prediction Dashboard" src="https://github.com/user-attachments/assets/422aed2f-8826-4dbc-a5e4-d4814fec50e8" />

### Prediction Result with SHAP Explanation & Feature Importance Analysis

<img width="1920" height="1080" alt="Screenshot (1952)" src="https://github.com/user-attachments/assets/83dab62b-a3c3-4d3a-b066-698b933b63ba" />

<img width="1920" height="1080" alt="Screenshot (1953)" src="https://github.com/user-attachments/assets/38db768e-7b9a-43d3-9bea-f40ce81a050b" />

---

## 📋 Project Overview

This project provides an **end-to-end data science solution** for predicting credit risk using machine learning. It covers the entire pipeline from data analysis and feature engineering to model training and deployment via a web interface.

### 🎯 Goal
Predict whether a credit application is **'Good'** (low risk) or **'Bad'** (high risk) using advanced machine learning techniques.

### 📊 Dataset
**German Credit Data** - A comprehensive dataset containing credit application information with historical risk labels for training and validation.

### 🔄 Key Steps in the Pipeline

1. **📊 Exploratory Data Analysis (EDA)**
   - Visualizing feature distributions
   - Analyzing relationships between features
   - Identifying patterns and anomalies
   - Understanding class imbalance

2. **🔨 Feature Engineering**
   - Handling missing values
   - Encoding categorical variables (Sex, Housing, Account Types)
   - Creating meaningful feature representations
   - Scaling and normalization for optimal model performance

3. **🤖 Modeling & Hyperparameter Tuning**
   - Experimenting with multiple tree-based algorithms:
     - Decision Tree
     - Random Forest
     - Extra Trees
     - XGBoost (Selected for production)
   - Hyperparameter optimization for best performance
   - Cross-validation for robust evaluation
   - Model comparison and selection

4. **🚀 Deployment**
   - Web application built with Streamlit
   - Real-time risk assessment capabilities
   - Interactive user interface for credit predictions
   - SHAP-based model explainability integration

---

## ✨ Key Features

- **🤖 Advanced ML Model**: XGBoost-powered credit risk classifier trained on German credit dataset
- **📊 Model Explainability**: SHAP (SHapley Additive exPlanations) integration for transparent predictions
- **🎯 Real-time Predictions**: Interactive Streamlit web interface for instant credit risk assessment
- **📈 Feature Analysis**: 
  - Individual prediction explanations via SHAP waterfall plots
  - Global feature importance visualization
  - Model performance metrics and insights
- **🔄 Fully Encoded**: Automatic handling of categorical variables (Sex, Housing, Account Types)
- **📦 Production Ready**: Serialized models and encoders for easy deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Installation & Run

```bash
# Clone and setup
git clone https://github.com/AliAoun/credit-risk-prediction-ml.git
cd credit-risk-prediction-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📖 Usage

1. **Enter applicant details**: Age, Sex, Job, Housing, Savings, Checking Account, Credit Amount, Duration
2. **Click "Predict Risk"**
3. **View results**: Risk classification (✅ GOOD / ❌ BAD) with SHAP explanations
   - Red features increase credit risk
   - Blue features decrease credit risk

---

## 📁 Project Structure

```
├── app.py                           # Streamlit web application
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── data/
│   └── german_credit_data.csv       # German credit dataset
├── models/
│   ├── xgboost_credit_model.pkl     # Trained XGBoost model
│   ├── sex_encoder.pkl
│   ├── housing_encoder.pkl
│   ├── saving_accounts_encoder.pkl
│   └── checking_account_encoder.pkl
└── notebook/
    └── credit_risk_model.ipynb      # Model development & training
```

---

## 🧠 Model Details

**Algorithm**: XGBoost (Extreme Gradient Boosting)  
**Dataset**: German Credit Data (1000+ samples)  
**Target**: Credit Risk (Good/Bad)  
**Features**: 8 engineered features  
**Explainability**: SHAP TreeExplainer  

See [credit_risk_model.ipynb](notebook/credit_risk_model.ipynb) for detailed model development, hyperparameter tuning, and performance metrics.

---

## 📦 Dependencies

Core packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`

---

## 🔄 Retraining

To retrain the model with new data:

1. Open `notebook/credit_risk_model.ipynb`
2. Run all notebook cells
3. Models save automatically to `models/`
4. Restart the Streamlit app

---

## 📊 Use Cases

- Automated credit approval workflows
- Portfolio risk assessment
- Fairness analysis in lending
- Explainable AI for credit decisions
- Regulatory compliance

---

## ⚠️ Important Notes

- Model trained on German credit data; performance may vary on other datasets
- Ensure data protection regulation compliance when deploying
- SHAP explanations most reliable near decision boundaries

---

## 🤝 Contributing

Fork the repo, create a feature branch, commit changes, and open a pull request.

---

## 📝 License

MIT License - see LICENSE file for details

---

## 📚 References

- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [SHAP Docs](https://shap.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [German Credit Dataset](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)

---

<div align="center">

⭐ **If you find this helpful, please star the repository!**

Built with ❤️ using Python, XGBoost, and SHAP

</div>
