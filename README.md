# Student Depression Prediction (Machine Learning)

Machine Learning project focused on **binary classification** to predict
the presence of depressive symptoms among students using demographic,
academic, and lifestyle features.

---

## 📊 Dataset

- Source: Kaggle Student Depression Dataset
- 18 features:
  - Numerical: age, academic pressure, CGPA, financial stress, etc.
  - Categorical: gender, sleep duration, dietary habits, suicidal thoughts, etc.
- Target variable:
  - `Depression` → 0 (no depression) / 1 (depression)

---

## 🔎 Project Workflow

1. **Data Preprocessing**
   - Handling categorical variables with One-Hot and Ordinal Encoding  
   - Train/validation/test split to avoid data leakage  

2. **Exploratory Data Analysis**
   - Correlation matrix and bivariate analysis  
   - Boxplots and bar charts vs target variable  

3. **Model Training**
   - Regression analysis  
   - Multiple classification models tested  
   - Best performance achieved with **SVM (RBF kernel)**  
   - Accuracy ≈ **85%**

4. **Evaluation**
   - Confusion matrix  
   - Cross-validation score distribution  
   - Statistical analysis of residuals and errors  

---

## 🛠️ Technologies Used

- Python  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## 🎯 Objective

Demonstrate a complete **end-to-end Machine Learning workflow**:

- Data preprocessing  
- Statistical analysis  
- Model training and tuning  
- Performance evaluation  

---

## 📌 Author

**Jean Baptiste Dindane**  
Computer Science for Management
