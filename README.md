# Student Depression Prediction (Binary Classification)

End-to-end Machine Learning project aimed at predicting the presence of depressive symptoms among students using demographic, academic, and lifestyle features.

---

## 📊 Dataset

- Source: Kaggle — Student Depression Dataset  
- URL: https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset  
- 18 features including:
  - **Numerical variables**: age, academic pressure, CGPA, financial stress, work/study hours, etc.
  - **Categorical variables**: gender, sleep duration, dietary habits, suicidal thoughts, profession, city, etc.
- **Target variable**:
  - `Depression` → 0 (no depression) / 1 (depression)

> The dataset is **not included** in this repository.  
> Download it from Kaggle and place it locally before running the notebook.

---

## 🔎 Machine Learning Workflow

### 1. Data Understanding & Preprocessing
- Inspection of dataset structure, variable types, and missing values  
- Separation of **numerical** and **categorical** features  
- Train/validation/test split to **avoid data leakage**  
- Encoding strategies:
  - **One-Hot Encoding** for nominal categorical variables  
  - **Ordinal Encoding** for ordered categorical variables  

### 2. Exploratory Data Analysis (EDA)
- Statistical summary of numerical variables  
- **Correlation matrix** to analyze linear relationships  
- **Bivariate analysis** between predictors and target variable  
- Visualization techniques:
  - Boxplots for numerical vs target  
  - Bar charts for categorical vs target  

Purpose: identify the most informative variables and detect potential patterns related to depression.

### 3. Model Training
- Evaluation of multiple classification approaches  
- Initial regression analysis to study relationships between strongly correlated variables  
- Training of **Support Vector Machine (SVM)** models with different kernels  
- Selection of the **best performing model** based on validation performance  

### 4. Model Evaluation
- Final testing on an unseen **test set**  
- Performance analysis using:
  - **Accuracy**
  - **Confusion matrix**
  - Cross-validation score distribution  
- Best model identified: **SVM with RBF kernel**.

---

## Results

- **Best model:** Support Vector Machine (RBF kernel)  
- **Accuracy:** ≈ **85%**  



---

## 🛠️ Technologies Used

- **Python**
- **NumPy & Pandas** for data manipulation  
- **Scikit-learn** for preprocessing, modeling, and evaluation  
- **Matplotlib & Seaborn** for data visualization  
- **Jupyter Notebook** for experimentation and analysis  

---

## ▶️ How to Run

```bash
git clone https://github.com/EpicNaje23/student-depression-prediction-ml.git
cd student-depression-prediction-ml
pip install -r requirements.txt
jupyter notebook
