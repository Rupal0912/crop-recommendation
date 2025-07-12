# 🌾 Crop Recommendation System using Machine Learning

This is a Machine Learning–based Crop Recommendation System designed to assist farmers and agricultural stakeholders in selecting the most suitable crop for cultivation based on soil and climate conditions.

The system uses supervised learning models to analyze factors like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall. It provides a Streamlit web app interface and supports multiple ML models like Random Forest, KNN, Logistic Regression, and XGBoost.

---

## 🚀 Features

- ✅ **Multi-model prediction**  
  Supports Random Forest, K-Nearest Neighbors (KNN), Logistic Regression, and XGBoost.

- 🔢 **Input via sliders**  
  Input features like Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall.

- 📊 **Dashboard includes:**
  - Feature importance (bar graph)
  - Model accuracy comparison
  - Confusion matrix (heatmap)

- 📅 **Crop Calendar**  
  Displays sowing/harvesting season for recommended crops.

- 🧪 **Soil Type Suggestion**  
  Dropdown with crop suggestions based on selected soil type.

- 📥 **Downloadable Reports**  
  Download prediction results in CSV format.

---

## 🛠️ Tech Stack

- **Programming Language**: Python 3.10+
- **Libraries**:  
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `streamlit`
- **IDE**: Visual Studio Code with Jupyter extension
- **Deployment**: Streamlit Cloud
- **Version Control**: Git & GitHub

---

## 📁 Project Structure

├── app/
│ ├── rf_model.pkl
│ ├── knn_model.pkl
│ ├── lr_model.pkl
│ ├── xgb_model.pkl
│ ├── scaler.pkl
│ ├── label_encoder.pkl
│ └── model_accuracies.pkl
├── data/
│ ├── crop_recommendation.csv
│ └── crop_recommendation_6600_realistic.csv
├── notebooks/
│ ├── EDA.ipynb
│ └── model_training.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

To run this project locally:

1. **Clone the repository**
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
If deployed on Streamlit Cloud or another hosting platform, add the link below:

🔗 Click here to try the live app

📈 Results
Model Accuracy Scores:

Model	Accuracy
Random Forest	98.2%
Logistic Regression	96.9%
K-Nearest Neighbors	97.3%
XGBoost	98.5%

📚 Dataset
Source: Kaggle – Crop Recommendation Dataset

Augmented from 2200 rows to 6600+ using noise injection techniques.

📌 Future Enhancements
🌦️ Live weather API integration

🌐 Multilingual interface for farmers

🧪 Fertilizer and pesticide suggestion modules

☁️ Cloud deployment with authentication

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

📄 License
This project is open-source and available under the MIT License.

🙌 Acknowledgements
Scikit-learn

XGBoost

Streamlit

Kaggle Dataset


