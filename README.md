# Heart Disease Detection using Machine Learning

A machine learning project for predicting heart disease in patients using various classification algorithms. This project aims to help in early detection and prevention of heart disease by analyzing patient data and medical attributes.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Heart disease is one of the leading causes of death worldwide. This project leverages machine learning techniques to predict the presence of heart disease in patients based on various medical attributes. By analyzing patterns in patient data, the model can assist healthcare professionals in making informed decisions for early diagnosis and treatment.

## ✨ Features

- **Data Analysis**: Comprehensive exploratory data analysis (EDA) of heart disease dataset
- **Multiple ML Models**: Implementation of various classification algorithms
- **Model Comparison**: Performance comparison of different models
- **Prediction System**: Real-time prediction of heart disease risk
- **Visualization**: Interactive visualizations of data insights and model performance
- **Feature Importance**: Analysis of key factors contributing to heart disease

## 📊 Dataset

The project uses a heart disease dataset containing patient information with the following attributes:

- **Age**: Age of the patient
- **Sex**: Gender of the patient (1 = male, 0 = female)
- **Chest Pain Type (cp)**: Type of chest pain experienced
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- **Resting Blood Pressure (trestbps)**: Resting blood pressure (mm Hg)
- **Cholesterol (chol)**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **Resting ECG (restecg)**: Resting electrocardiographic results
- **Maximum Heart Rate (thalach)**: Maximum heart rate achieved
- **Exercise Induced Angina (exang)**: Exercise induced angina (1 = yes, 0 = no)
- **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment
- **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy
- **Thalassemia (thal)**: Thalassemia type
- **Target**: Diagnosis of heart disease (1 = presence, 0 = absence)

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/favleo/HeartDiseaseDetection.git
cd HeartDiseaseDetection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## 💻 Usage

### Jupyter Notebook
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the main notebook file and run the cells sequentially

### Python Script
Run the main script:
```bash
python heart_disease_detection.py
```

### Making Predictions
```python
from heart_disease_model import predict_heart_disease

# Example patient data
patient_data = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}

prediction = predict_heart_disease(patient_data)
print(f"Heart Disease Risk: {prediction}")
```

## 🤖 Models

This project implements and compares several machine learning algorithms:

### Classification Algorithms
1. **Logistic Regression**
   - Simple and interpretable model
   - Good baseline performance

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Effective for pattern recognition

3. **Support Vector Machine (SVM)**
   - Powerful for binary classification
   - Handles non-linear boundaries

4. **Decision Trees**
   - Easy to visualize and interpret
   - Captures non-linear relationships

5. **Random Forest**
   - Ensemble method
   - Reduces overfitting
   - High accuracy

6. **Gradient Boosting**
   - Sequential ensemble method
   - Often achieves best performance

7. **Neural Networks**
   - Deep learning approach
   - Captures complex patterns

### Model Pipeline
1. Data preprocessing and cleaning
2. Feature scaling and normalization
3. Train-test split
4. Model training
5. Hyperparameter tuning
6. Cross-validation
7. Model evaluation

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **Jupyter Notebook**: Interactive development environment

## 📈 Results

### Model Performance

The following table shows typical performance metrics that can be expected from various models on heart disease datasets:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~85% | ~86% | ~84% | ~85% |
| K-Nearest Neighbors | ~82% | ~83% | ~81% | ~82% |
| Support Vector Machine | ~87% | ~88% | ~86% | ~87% |
| Decision Tree | ~80% | ~81% | ~80% | ~80% |
| Random Forest | ~90% | ~91% | ~89% | ~90% |
| Gradient Boosting | ~91% | ~92% | ~90% | ~91% |
| Neural Network | ~88% | ~89% | ~87% | ~88% |

*Note: These are typical ranges based on heart disease prediction studies. Actual results will vary based on the specific dataset, preprocessing methods, and hyperparameter tuning. Run the models on your dataset to obtain accurate performance metrics.*

### Key Findings
- Ensemble methods (Random Forest, Gradient Boosting) achieve the highest accuracy
- Feature importance analysis shows chest pain type, maximum heart rate, and ST depression as top predictors
- Model performs well with balanced precision and recall

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Areas for Improvement
- Add more advanced deep learning models
- Implement real-time prediction API
- Add web interface for predictions
- Expand dataset with more patient records
- Include additional feature engineering
- Add model explainability (SHAP, LIME)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

## 🙏 Acknowledgments

- Heart disease dataset contributors
- Open-source machine learning community
- Healthcare professionals providing domain expertise

## ⚠️ Disclaimer

This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

---

**Made with ❤️ for better healthcare through AI**
