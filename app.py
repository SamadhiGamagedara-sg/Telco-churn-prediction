import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
preprocessor = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/churn_model_rf.pkl")
le = joblib.load("artifacts/label_encoder.pkl")

st.title("Telco Customer Churn Predictor")
st.write("Enter customer details to predict churn risk.")

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure_bins = st.selectbox("Tenure Group", ["Newer", "Medium", "Long-term"])

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# --- Predict ---
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_bins": tenure_bins,
    }])

    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)[0]
    probability = model.predict_proba(X_transformed)[0][1]

    label = le.inverse_transform([prediction])[0]

    if label == "Yes":
        st.error(f"⚠️ High churn risk — predicted: {label} (probability: {probability:.1%})")
    else:
        st.success(f"✅ Low churn risk — predicted: {label} (probability of churn: {probability:.1%})")