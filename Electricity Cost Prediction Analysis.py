# %% [markdown]
# # Importing Libraries

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    accuracy_score,r2_score,classification_report,
    mean_squared_error,mean_absolute_error,root_mean_squared_error
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# %%
train=pd.read_csv(r"C:\Users\user\Desktop\Kaggle Datasets\Electricity Prediction Dataset\Dataset\Train.csv")
test=pd.read_csv(r"C:\Users\user\Desktop\Kaggle Datasets\Electricity Prediction Dataset\Dataset\Test.csv")
submission=pd.read_csv(r"C:\Users\user\Desktop\Kaggle Datasets\Electricity Prediction Dataset\Dataset\Submission.csv")

# %% [markdown]
# # Data Info 

# %%
train.head()

# %%
train.shape

# %%
train.isna().sum()
train.duplicated().sum()

# %%
test.head()

# %%
test.shape

# %%
test.isna().sum()

# %%
test.duplicated().sum()

# %%
submission.head()

# %% [markdown]
# # Data Handling & Cleaning 

# %%
train.head()

# %%
test.head()

# %%
# Step 1: Compute mode from the training set
mode_value = train["Electricity_Cost"].mode()[0]  # [0] to extract the actual mode value

# Step 2: Fill missing values in the test set
test["Electricity_Cost"] = test["Electricity_Cost"].fillna(mode_value)

# %%
test.head()
test.isna().sum()
test.duplicated().sum()

# %% [markdown]
# # EDA & Machine Learning

# %%
train.columns

# %%
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=train[col], palette="Set1")
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


# %%
train["Electricity_Cost"].describe().plot(kind="barh",figsize=(12,6),title="Electricity Cost Details")

# %%
le=LabelEncoder()
train["Structure_Type "]=le.fit_transform(train["Structure_Type "])
train = pd.get_dummies(train, columns=["Site_Area"], prefix="Site_Area")

# %%
bool_cols = train.select_dtypes(include='bool').columns
train[bool_cols] = train[bool_cols].astype(int)

# %%
# Label encode 'Structure_Type'
test["Structure_Type "] = le.transform(test["Structure_Type "])  # Use transform, not fit_transform

# One-hot encode 'Site_Area'
test = pd.get_dummies(test, columns=["Site_Area"], prefix="Site_Area")


# %%
bool_colsts = test.select_dtypes(include='bool').columns
test[bool_colsts] = test[bool_colsts].astype(int)

# %%
test.head()

# %%
train.head()

# %% [markdown]
# ## Linear Regression Model 

# %%
x_train=train.drop(columns=["Electricity_Cost"])
y_train=train["Electricity_Cost"]
from sklearn.preprocessing import StandardScaler
test_input = test.drop(columns=["Electricity_Cost"], errors="ignore")
# Step 1: Initialize scaler
scaler = StandardScaler()

# Step 2: Fit scaler on training data and transform both train and test
x_train_scaled = scaler.fit_transform(x_train)
test_input_scaled = scaler.transform(test_input)  # test_input should match x_train columns

# Optional: Convert back to DataFrame (preserves column names)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
test_input_scaled = pd.DataFrame(test_input_scaled, columns=x_train.columns, index=test_input.index)
model_lr=LinearRegression()
model_lr.fit(x_train_scaled, y_train)
predictions = model_lr.predict(test_input_scaled)


# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the scaled test data
predictions_scaled = model_lr.predict(test_input_scaled)

# Grab the actual target values
y_true = test["Electricity_Cost"].values

# Evaluate
mae = mean_absolute_error(y_true, predictions_scaled)
mse = mean_squared_error(y_true, predictions_scaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, predictions_scaled)

# Display results
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# %% [markdown]
# ## Random Forest Regressor Model 

# %%
model_rf=RandomForestRegressor()
model_rf.fit(x_train,y_train)
prediction_rf=model_rf.predict(test_input)

# %%
# Ground truth (actual values)
y_true = test["Electricity_Cost"]

# Predictions
y_pred = prediction_rf

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Display results
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

model_hgb = HistGradientBoostingRegressor()
model_hgb.fit(x_train, y_train)
prediction_hgb = model_hgb.predict(test_input)

# Evaluate it
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = test["Electricity_Cost"]
mae = mean_absolute_error(y_true, prediction_hgb)

r2 = r2_score(y_true, prediction_hgb)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
rmse = mean_squared_error(y_true, prediction_hgb)


# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1️⃣ Initialize the model
model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# 2️⃣ Fit the model on scaled training data
model_xgb.fit(x_train_scaled, y_train)

# 3️⃣ Predict on scaled test data
predictions_xgb = model_xgb.predict(test_input_scaled)

# 4️⃣ Evaluate
y_true = test["Electricity_Cost"].values
mae = mean_absolute_error(y_true, predictions_xgb)
rmse = np.sqrt(mean_squared_error(y_true, predictions_xgb))
r2 = r2_score(y_true, predictions_xgb)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



