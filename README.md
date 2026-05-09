# 💳 Credit Risk Prediction ML

A sophisticated machine learning application for predicting credit risk using XGBoost with model explainability powered by SHAP. Built with a modern Streamlit web interface for real-time predictions and intuitive feature analysis.

---

## 🎨 Application Screenshots

### Main Dashboard
![Credit Risk Prediction Dashboard](https://via.placeholder.com/1200x400/4A90E2/FFFFFF?text=Main+Dashboard)

### Prediction Result with SHAP Explanation
![SHAP Waterfall Plot](https://via.placeholder.com/1200x400/7B68EE/FFFFFF?text=SHAP+Explanation)

### Feature Importance Analysis
![Feature Importance](https://via.placeholder.com/1200x400/50C878/FFFFFF?text=Feature+Importance+Chart)

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
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-risk-prediction-ml.git
   cd credit-risk-prediction-ml
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Launch the Streamlit web app:**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step-by-Step Prediction

1. **Enter Applicant Information**:
   - **Age**: Applicant's age (18-100)
   - **Sex**: Male or Female
   - **Job**: Job classification (0-3 scale)
   - **Housing**: Housing type (own, rent, free)
   - **Saving Accounts**: Savings level (little, moderate, rich, quite rich)
   - **Checking Account**: Checking account status (little, moderate, rich)
   - **Credit Amount**: Loan amount requested (€)
   - **Duration**: Loan duration in months

2. **Click "Predict Risk"** to generate predictions

3. **View Results**:
   - **Risk Classification**: ✅ GOOD or ❌ BAD
   - **SHAP Waterfall Plot**: See how each feature contributed to the prediction
   - **Global Feature Importance**: Understand which features matter most overall

### Understanding SHAP Explanations

- **Red features**: Increase credit risk (push toward BAD)
- **Blue features**: Decrease credit risk (push toward GOOD)
- **Base value**: Model's average prediction
- **Feature contribution**: Shows the impact of each feature on the final prediction

---

## 📁 Project Structure

```
credit-risk-prediction-ml/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── data/
│   └── german_credit_data.csv          # German credit dataset
├── models/
│   ├── xgboost_credit_model.pkl        # Trained XGBoost model
│   ├── sex_encoder.pkl                 # Categorical encoder (Sex)
│   ├── housing_encoder.pkl             # Categorical encoder (Housing)
│   ├── saving_accounts_encoder.pkl     # Categorical encoder (Savings)
│   └── checking_account_encoder.pkl    # Categorical encoder (Checking)
└── notebook/
    └── credit_risk_model.ipynb         # Jupyter notebook (Model development & training)
```

---

## 🧠 Model Details

### Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Dataset**: German Credit Dataset (1000+ samples)
- **Target Variable**: Credit Risk (Good/Bad)
- **Features**: 8 engineered features
- **Explainability**: SHAP TreeExplainer

### Model Capabilities
- **Binary Classification**: Predicts GOOD or BAD credit risk
- **Probability Scores**: Outputs confidence levels for each prediction
- **Feature Interactions**: Captures complex relationships between features
- **Instance-level Explanations**: SHAP values for individual predictions
- **Global Interpretability**: Feature importance across entire dataset

### Training Details
Refer to [credit_risk_model.ipynb](notebook/credit_risk_model.ipynb) for:
- Data exploration and preprocessing
- Feature engineering techniques
- Model hyperparameter tuning
- Cross-validation results
- Performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Machine learning utilities & preprocessing |
| `xgboost` | Gradient boosting models |
| `shap` | Model explainability |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical visualization |
| `joblib` | Model persistence |
| `jupyter` | Notebook development |

---

## 🛠️ Technologies & Stack

### Core Technologies
- **Language**: Python 3.8+
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Explainability**: SHAP
- **Web Framework**: Streamlit
- **Model Persistence**: joblib
- **Development Environment**: Jupyter Notebook

### Architecture
```
Data → EDA & Analysis → Feature Engineering → Model Training → Evaluation → Deployment
                                                                              ↓
                                                                        Streamlit App
                                                                        (SHAP Explainability)
```

---

## 🔧 Configuration

### Model Files
All trained models and encoders are pre-loaded from the `models/` directory. To retrain:

1. Open [credit_risk_model.ipynb](notebook/credit_risk_model.ipynb)
2. Run all cells to regenerate models
3. Models will be saved to `models/` directory
4. Restart the Streamlit app

### Streamlit Configuration
Customize the Streamlit app settings by modifying `app.py`:
- Page title: `page_title="Credit Risk Prediction"`
- Page icon: `page_icon="💳"`
- Layout: `layout="centered"`

---

## 🎯 Performance Metrics

The model achieves competitive performance on the German Credit Dataset:
- Precision and Recall for both GOOD and BAD cases
- ROC-AUC score for model discrimination ability
- Cross-validation results for robustness

*See notebook for detailed metrics and model evaluation.*

---

## 💡 How SHAP Works

SHAP (SHapley Additive exPlanations) provides game-theoretic interpretations of model predictions:

1. **Waterfall Plots**: Shows the contribution of each feature to push the prediction from base value to final output
2. **Feature Importance**: Ranks features by their average absolute SHAP values across all predictions
3. **Fairness**: Built on Shapley values from cooperative game theory - ensures fair feature attribution

---

## 🛠️ Development

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dev dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Launch Jupyter notebook for exploration
jupyter notebook notebook/credit_risk_model.ipynb
```

### Model Training
The model is pre-trained. To retrain with new data or parameters:
1. Update `notebook/credit_risk_model.ipynb`
2. Run all notebook cells
3. Models will be automatically saved to `models/`

---

## 📊 Use Cases

- **Bank Credit Decisions**: Automated credit approval workflow
- **Risk Management**: Portfolio risk assessment
- **Fairness Analysis**: Identify potential biases in lending decisions
- **Customer Communication**: Explain why credit was approved/denied using SHAP
- **Regulatory Compliance**: Explainable AI for credit decisions

---

## ⚠️ Important Notes

- **Data Privacy**: This model is trained on public German Credit Dataset. Ensure compliance with data protection regulations when deploying
- **Model Limitations**: Model trained on German credit data - performance may vary on other datasets
- **Feature Ranges**: Input validation is implemented for age (18-100), job (0-3), and credit amount
- **Explainability**: SHAP explanations are most reliable for predictions near the decision boundary

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution:
- Additional feature engineering techniques
- Model hyperparameter optimization
- Performance metrics improvements
- UI/UX enhancements for Streamlit app
- Documentation improvements
- Unit tests and integration tests
- Deployment examples (Docker, AWS, Azure)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📚 References

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **SHAP Documentation**: https://shap.readthedocs.io/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **German Credit Dataset**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- **Issues**: Open an issue on GitHub
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]

---

## 🎓 Learning Resources

New to machine learning or SHAP? Check these resources:
- [Introduction to XGBoost](https://towardsdatascience.com/xgboost-a-detailed-explanation-32cf0f179419)
- [Understanding SHAP](https://towardsdatascience.com/shap-explain-any-machine-learning-model-in-python-24207bbb1d27)
- [Streamlit Tutorial](https://streamlit.io/docs)
- [Explainable AI Guide](https://christophm.github.io/interpretable-ml-book/)

---

<div align="center">

### ⭐ If you find this project helpful, please consider starring it!

**Built with ❤️ using Python, XGBoost, and SHAP**

</div>
