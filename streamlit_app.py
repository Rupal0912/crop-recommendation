
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO
from sklearn.metrics import confusion_matrix

# Load models and accuracy scores
model_dict = {
    "Random Forest": joblib.load("app/rf_model.pkl"),
    "Logistic Regression": joblib.load("app/lr_model.pkl"),
    "KNN": joblib.load("app/knn_model.pkl"),
    "XGBoost": joblib.load("app/xgb_model.pkl"),
}
model_scores = joblib.load("app/model_accuracies.pkl")

le = joblib.load("app/label_encoder.pkl")

# Static info
crop_info = {
    "rice": "Needs high humidity, temp ~25-30°C, heavy rainfall.",
    "wheat": "Requires cool growing season and bright sunshine.",
    "maize": "Grows well in loamy soil, moderate water needs.",
    "cotton": "Needs light soil and high temperature.",
    "millet": "Tolerant to drought and poor soils.",
    "ground nut": "Grows in sandy loam soil and warm climate.",
}

soil_types = {
    "Loamy": ["Rice", "Wheat", "Sugarcane"],
    "Clay": ["Paddy", "Jute", "Sugarcane"],
    "Sandy": ["Groundnut", "Millets", "Cotton"],
    "Silty": ["Potato", "Soybean", "Tomato"]
}

crop_calendar = {
    "rice": "Sow: June–July, Harvest: Oct–Nov",
    "wheat": "Sow: Nov–Dec, Harvest: Mar–Apr",
    "maize": "Sow: May–June, Harvest: Sept–Oct",
    "cotton": "Sow: April–May, Harvest: Oct–Nov",
    "millet": "Sow: June–July, Harvest: Sept–Oct",
    "ground nut": "Sow: June, Harvest: Oct"
}

# Streamlit config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.title("🌱 Smart Crop Recommendation System")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Dashboard", "🌿 Soil", "📅 Crop Calendar"])

# -------------------- TAB 1: Prediction --------------------
with tab1:
    selected_model_name = st.selectbox("Choose Model", list(model_dict.keys()))
    model = model_dict[selected_model_name]
    st.metric(label="🎯 Model Accuracy", value=f"{model_scores[selected_model_name]*100:.2f}%")

    st.subheader("🔍 Predict the Best Crop")

    N = st.slider("Nitrogen", 0, 140, 80)
    P = st.slider("Phosphorus", 5, 145, 40)
    K = st.slider("Potassium", 5, 205, 40)
    temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
    ph = st.slider("pH", 3.5, 10.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

    if st.button("🌿 Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        predicted_crop = le.inverse_transform(prediction)[0]

        st.success(f"🚀 Recommended Crop: **{predicted_crop.capitalize()}**")
        if predicted_crop in crop_info:
            st.info(f"ℹ️ {crop_info[predicted_crop]}")
        if predicted_crop in crop_calendar:
            st.info(f"📅 {crop_calendar[predicted_crop]}")

        # Downloadable report
        result_df = pd.DataFrame({
            "Nitrogen": [N], "Phosphorus": [P], "Potassium": [K],
            "Temperature": [temperature], "Humidity": [humidity],
            "pH": [ph], "Rainfall": [rainfall],
            "Predicted Crop": [predicted_crop.capitalize()],
            "Model": [selected_model_name]
        })
        csv_buffer = StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Prediction Report", data=csv_buffer.getvalue(),
                           file_name="crop_prediction_report.csv", mime="text/csv")

# -------------------- TAB 2: Dashboard --------------------
with tab2:
    st.header("📊 Data Dashboard")

    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots()
        sns.barplot(x=model.feature_importances_, y=['N','P','K','Temp','Humidity','pH','Rainfall'], palette='viridis', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    st.subheader("Model Accuracy Comparison")
    accuracy_data = pd.DataFrame({
        "Model": list(model_scores.keys()),
        "Accuracy": [round(acc * 100, 2) for acc in model_scores.values()]
    })
    fig2, ax2 = plt.subplots()
    sns.barplot(data=accuracy_data, x="Model", y="Accuracy", palette="Set2", ax=ax2)
    ax2.set_ylim(90, 100)
    st.pyplot(fig2)

    st.subheader("Confusion Matrix (Simulated)")
    # Simulate test set
    X_sample = np.random.rand(30, 7) * 100
    y_sample = np.random.randint(0, len(le.classes_), 30)
    y_pred = model.predict(X_sample)
    cm = confusion_matrix(y_sample, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=False, fmt='d', cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (Simulated)")
    st.pyplot(fig_cm)

# -------------------- TAB 3: Soil --------------------
with tab3:
    st.header("🌿 Soil Type Info")
    soil = st.selectbox("Select Soil Type", list(soil_types.keys()))
    st.info(f"✅ Suitable Crops: {', '.join(soil_types[soil])}")

# -------------------- TAB 4: Crop Calendar --------------------
with tab4:
    st.header("📅 Crop Calendar")
    st.write(pd.DataFrame.from_dict(crop_calendar, orient='index', columns=["Sowing & Harvesting"]))
