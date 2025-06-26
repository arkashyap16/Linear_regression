# 🏡 Housing Price Prediction - Linear Regression (Task 3)

The objective of this task is to implement **Simple and Multiple Linear Regression** using the Housing Price Prediction dataset and understand key evaluation metrics and model behavior.
---

## 🎯 Task Objective

> Implement and analyze **Linear Regression** using a real-world housing dataset to predict property prices based on various features like area, location, furnishing status, and more.

---

## 📦 Dataset

- **Name**: Housing Price Prediction Dataset  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- **File Used**: `Housing.csv`

---

## 🧰 Tools & Libraries Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔄 Workflow Summary

### 1. **Data Preprocessing**
- Checked for missing values and data types
- Encoded categorical features using:
  - Binary Mapping for `yes`/`no` columns (e.g., `mainroad`, `guestroom`)
  - One-hot encoding for `furnishingstatus`

### 2. **Model Building**
- Defined `X` (features) and `y` (target: price)
- Performed **train-test split** (80/20)
- Trained **LinearRegression** model using `sklearn`

### 3. **Model Evaluation**
Used the following metrics:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

### 4. **Visualization**
- Plotted **Actual vs Predicted** price comparison
- Regression diagnostic plots

- **Here are some screenshots of the code output and visulaization**
![Screenshot 2025-06-26 202015](https://github.com/user-attachments/assets/36c99934-b9b3-4815-8aaa-bc34e44b58d3)
![Screenshot 2025-06-26 201958](https://github.com/user-attachments/assets/6cdb6776-a298-462b-ad41-529919dcabce)
![Screenshot 2025-06-26 201911](https://github.com/user-attachments/assets/4a7e328a-f5ea-4c60-90f7-41838ff1f535)

---------------------------------**Have a good code**--------------------------------------

- 
