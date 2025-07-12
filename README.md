🌱 Crop Recommendation System using Machine Learning
This project is a Machine Learning-based Crop Recommendation System designed to assist farmers and agricultural stakeholders in selecting the most suitable crop for cultivation, based on soil and environmental parameters.

The system uses multiple supervised learning models to analyze factors such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall. It provides real-time predictions via a Streamlit web app interface and supports multiple ML models like Random Forest, KNN, Logistic Regression, and XGBoost.

🚀 Features
✅ Multi-model prediction (Random Forest, KNN, Logistic Regression, XGBoost)

🌡️ Input parameters via sliders (N, P, K, Temperature, Humidity, pH, Rainfall)

📊 Dashboard with:

Feature importance

Model accuracy comparison

Confusion matrix (heatmap)

🌾 Crop Calendar for sowing/harvest season

🧪 Soil type suggestion module

📥 Downloadable prediction reports (CSV format)

🧠 Model accuracy display

☁️ Deployable via Streamlit Cloud

🖼️ App Interface Preview
(Add screenshots or GIFs here: one for Prediction tab, one for Dashboard)

📁 Project Structure
kotlin
Copy
Edit
crop-recommendation/
├── app/
│   ├── rf_model.pkl
│   ├── knn_model.pkl
│   ├── lr_model.pkl
│   ├── xgb_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── model_accuracies.pkl
├── data/
│   ├── crop_recommendation.csv
│   └── crop_recommendation_6600_realistic.csv
├── notebooks/
│   └── model_training.ipynb
├── streamlit_app.py
├── README.md
└── requirements.txt
🔧 Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/<your-username>/crop-recommendation.git
cd crop-recommendation
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit App

bash
Copy
Edit
streamlit run streamlit_app.py
Make sure the .pkl model files and datasets are correctly placed inside their respective folders (/app, /data).

📊 Machine Learning Models
Model	Accuracy
Random Forest	98.2%
Logistic Regression	96.9%
K-Nearest Neighbors	97.3%
XGBoost	98.5%

All models were trained using GridSearchCV and evaluated using standard classification metrics.

🔮 Future Enhancements
🌐 Live Weather API integration for real-time weather-based prediction

🗣️ Multilingual support for regional farmers

📱 Mobile app version

💊 Fertilizer & pest control suggestions

☁️ Cloud database for storing user history

📚 References
Kaggle Dataset

Scikit-learn Documentation

XGBoost Documentation

Streamlit Documentation

🙌 Acknowledgements
This project was developed as part of an academic training initiative to explore real-world AI applications in agriculture. Special thanks to mentors and faculty for their guidance and support.
