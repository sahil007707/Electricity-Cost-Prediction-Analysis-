# ⚡ Electricity Cost Prediction: Advanced EDA & Machine Learning

![Electricity](https://img.shields.io/badge/Machine%20Learning-Electricity%20Cost%20Prediction-blueviolet?style=for-the-badge&logo=scikit-learn)
  
## 📌 Project Overview

This project leverages advanced data analysis and machine learning techniques to predict **electricity cost** based on a variety of structural and site-related features. The goal is to build robust regression models capable of providing accurate and reliable estimates of electricity costs, potentially aiding organizations or residential consumers in forecasting their energy expenses.

---

## 🗂️ Dataset

- **Train.csv**: Used for training and validation  
- **Test.csv**: Used for testing and final submission  
- **Submission.csv**: Submission template  

Key columns:
- `Structure_Type` : Categorical data on the type of structure  
- `Site_Area` : One-hot encoded site areas  
- `Electricity_Cost`: Target variable for regression  

---

## 🚀 Approach

### 1️⃣ Exploratory Data Analysis (EDA)

- Missing value analysis & handling  
- Outlier detection via boxplots  
- Distribution analysis of numerical features  
- Correlation matrix visualization  
- Label encoding & one-hot encoding of categorical variables

### 2️⃣ Feature Engineering

- Encoded categorical features with Label Encoding and One-Hot Encoding  
- Conversion of boolean columns to integers  
- Standard scaling for linear models

### 3️⃣ Model Building

- **Linear Regression**  
- **Random Forest Regressor**  
- **XGBoost Regressor**  
- **HistGradientBoostingRegressor**  
- **KNeighbors Regressor**

Each model was evaluated using:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score

with cross-validation for consistent evaluation.

---

## 🧩 Challenges Faced

- Handling inconsistent features between train and test sets  
- Preventing data leakage from target variable  
- Managing scaling across multiple models  
- Addressing potential feature mismatches after encoding

---

## 📊 Results

After tuning and validation, the best-performing models achieved reasonable RMSE and R² scores, though further hyperparameter tuning and domain-driven feature engineering could improve performance even more.  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib & Seaborn  
- Plotly  

---

## 🤝 Contributing

Feel free to open issues or submit pull requests if you want to extend the feature engineering, add ensembling, or optimize hyperparameters further!  

---

## 📄 License

This project is released under the [MIT License](LICENSE).  

---

## 🙌 Acknowledgments

Special thanks to the open-source community for providing robust tools and frameworks to make this project possible.

---

> *Crafted with 💙 by [Md Sahil Islam]*
> sahil07707- Github

