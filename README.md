**🩺 COVID-19 Chest X-Ray Classification**

This project builds a deep learning model to classify chest X-ray images into:

COVID-19

Viral Pneumonia

Normal

The goal is to support clinical decision-making by providing fast and interpretable predictions.

**✅ Features**

Transfer Learning using ResNet50

Grad-CAM visualization to show model attention regions

Streamlit Web App for image upload and prediction

Model can be deployed to cloud / hospital systems

**📂 Dataset**

Kaggle Dataset: COVID-19 Radiography Database
Contains X-ray images for the 3 classes.
Images were resized and normalized during preprocessing.

**🧠 Model Workflow**

Image Preprocessing

Transfer Learning (ResNet50 Fine-Tuned)

Training & Model Evaluation

Explainability using Grad-CAM

Deployment as a Streamlit Web App

**🚀 Running the Streamlit App**
streamlit run app.py


Upload an X-ray → Model predicts → Grad-CAM heatmap is shown.

**🧪 Evaluation Metrics**

Accuracy

Precision / Recall / F1-score

Confusion Matrix

Visual Explainability (Grad-CAM)

**🔍 Why Explainability?**

Grad-CAM highlights infected lung areas the model used for classification.
This increases clinical trust and ensures transparent AI in healthcare.
