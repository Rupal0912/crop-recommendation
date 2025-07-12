# 🌾 Crop Recommendation System using Machine Learning

This project is a **Machine Learning–based Crop Recommendation System** designed to assist farmers and agricultural stakeholders in selecting the most suitable crop for cultivation based on soil and climate parameters.

It uses multiple supervised learning models like **Random Forest, KNN, Logistic Regression**, and **XGBoost**, and is deployed using **Streamlit** for interactive web access.

---

## 🚀 Features

- ✅ Multi-model prediction (Random Forest, KNN, Logistic Regression, XGBoost)
- 🔧 Input via sliders for:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall
- 📊 Dashboard includes:
  - Feature Importance
  - Model Accuracy Comparison
  - Confusion Matrix (heatmap)
- 📅 Crop Calendar (Sowing & Harvesting guidance)
- 🌱 Soil Type Suggestion Module
- 📥 Downloadable prediction reports (CSV format)

---

## 🧠 Machine Learning Models

| Model                | Accuracy   |
|---------------------|------------|
| Random Forest        | 98.2%      |
| Logistic Regression  | 96.9%      |
| K-Nearest Neighbors  | 97.3%      |
| XGBoost              | 98.5%      |

All models were trained on a dataset of 6600+ samples (augmented from 2200 using noise injection) and saved as `.pkl` files using `joblib`.

---

crop-recommendation/
├── app/
│ ├── rf_model.pkl
│ ├── lr_model.pkl
│ ├── knn_model.pkl
│ ├── xgb_model.pkl
│ ├── model_accuracies.pkl
│ ├── scaler.pkl
│ └── label_encoder.pkl
├── data/
│ ├── crop_recommendation.csv
│ └── crop_recommendation_6600_realistic.csv
├── notebooks/
│ ├── EDA.ipynb
│ └── model_training.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ How to Run This Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/rupal0912/crop-recommendation.git
cd crop-recommendation
2. Install the required libraries
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit app
bash
Copy
Edit
streamlit run streamlit_app.py
🌐 Live Demo (Optional)
👉 Click here to try the live app

📊 Dataset Source
Kaggle – Crop Recommendation Dataset

Augmented from 2200 to 6600+ rows using noise injection for better generalization

🔮 Future Enhancements
🌦️ Live weather API integration

🌍 Multilingual interface for wider farmer accessibility

🧪 Fertilizer and pest management suggestions

☁️ Cloud deployment with login support

🙌 Acknowledgements
Scikit-learn

XGBoost

Streamlit

Kaggle Dataset

📄 License
This project is licensed under the MIT License.


