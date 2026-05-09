import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# LOAD MODEL

model = joblib.load(
    "models/xgboost_credit_model.pkl"
)

explainer = shap.Explainer(model)

# LOAD ENCODERS

encoders = {

    "Sex":
    joblib.load("models/sex_encoder.pkl"),

    "Housing":
    joblib.load("models/housing_encoder.pkl"),

    "Saving accounts":
    joblib.load("models/saving_accounts_encoder.pkl"),

    "Checking account":
    joblib.load("models/checking_account_encoder.pkl")
}

# PAGE CONFIG

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# TITLE

st.title("💳 Credit Risk Prediction App")

st.write(
    "Enter applicant information to predict credit risk."
)

# USER INPUTS

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.number_input(
    "Job (0-3)",
    min_value=0,
    max_value=3,
    value=1
)

housing = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "rich", "quite rich"]
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich"]
)

credit_amount = st.number_input(
    "Credit Amount",
    min_value=0,
    value=1000
)

duration = st.number_input(
    "Duration (Months)",
    min_value=1,
    value=12
)

# PREDICT BUTTON

if st.button("Predict Risk"):

    input_df = pd.DataFrame({

        "Age":[age],

        "Sex":[
            encoders["Sex"].transform([str(sex)])[0]
        ],

        "Job":[job],

        "Housing":[
            encoders["Housing"].transform([str(housing)])[0]
        ],

        "Saving accounts":[
            encoders["Saving accounts"].transform(
                [str(saving_accounts)]
            )[0]
        ],

        "Checking account":[
            encoders["Checking account"].transform(
                [str(checking_account)]
            )[0]
        ],

        "Credit amount":[credit_amount],

        "Duration":[duration]
    })

    prediction = model.predict(input_df)[0]

    shap_values = explainer(input_df)

    if prediction == 1:

        st.success(
            "✅ Predicted Credit Risk: GOOD"
        )

    else:

        st.error(
            "❌ Predicted Credit Risk: BAD"
        )

    st.subheader("SHAP Explanation")

    plt.clf()

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(fig)

    st.subheader("Global Feature Importance")

    plt.clf()

    fig2, ax2 = plt.subplots()

    shap.plots.bar(
        shap_values,
        show=False
    )

    st.pyplot(fig2)