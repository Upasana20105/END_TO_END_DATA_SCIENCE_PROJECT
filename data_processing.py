import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib # For saving the model and preprocessor

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Please download 'train.csv' and 'test.csv' from Kaggle and place them in the same directory as this script.")
    exit()

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# --- EDA (Illustrative Examples) ---
# Display basic info
print("\nTrain Info:")
train_df.info()

print("\nTest Info:")
test_df.info()

# Check for missing values
print("\nMissing values in Train data:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0].sort_values(ascending=False))

print("\nMissing values in Test data:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0].sort_values(ascending=False))

# Visualize SalePrice distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap (numerical features)
plt.figure(figsize=(12, 10))
sns.heatmap(train_df.select_dtypes(include=np.number).corr(), cmap='coolwarm', annot=False, fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --- Preprocessing Steps ---
# Separate target variable
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Combine train and test data for consistent preprocessing (handling columns that might only exist in one)
# Keep track of original train/test split index
train_ids = X['Id']
test_ids = test_df['Id']
X_combined = pd.concat([X.drop('Id', axis=1), test_df.drop('Id', axis=1)], ignore_index=True)

# Identify numerical and categorical features
global_numerical_features = X_combined.select_dtypes(include=np.number).columns.tolist()
global_categorical_features = X_combined.select_dtypes(include='object').columns.tolist()

# Define preprocessing pipelines for numerical and categorical features
# Impute missing numerical values with the mean
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Impute missing categorical values with the most frequent value and then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for new categories in test set
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, global_numerical_features),
        ('cat', categorical_transformer, global_categorical_features)
    ])

# Fit and transform the combined data
X_processed = preprocessor.fit_transform(X_combined)

# Convert back to DataFrame (optional, but good for understanding)
# This step is complex due to OneHotEncoder creating many new columns.
# For deployment, the pipeline directly works with processed numpy array.
# For now, let's keep it as a sparse matrix or numpy array for simplicity in model training.

# Split back into training and testing sets
X_train_processed = X_processed[:len(train_df)]
X_test_processed = X_processed[len(train_df):]

# Optional: Log transform the target variable to handle skewness
y_log = np.log1p(y)

print("\nPreprocessing complete. Shapes after preprocessing:")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape:", X_test_processed.shape)
print("y_log shape:", y_log.shape)
# --- Model Training ---
# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y_log, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model (on log-transformed scale)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)

print(f"\nModel Performance on Validation Set (log-transformed):")
print(f"RMSE: {rmse_val:.4f}")
print(f"R-squared: {r2_val:.4f}")

# Convert predictions back to original scale for better interpretability
y_pred_val_original = np.expm1(y_pred_val)
y_val_original = np.expm1(y_val)
rmse_val_original = np.sqrt(mean_squared_error(y_val_original, y_pred_val_original))
print(f"RMSE on Original Scale: {rmse_val_original:.2f}")

# --- Save the trained model and the preprocessor ---
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(preprocessor, 'house_price_preprocessor.pkl')
joblib.dump(global_numerical_features, 'numerical_features.pkl')
joblib.dump(global_categorical_features, 'categorical_features.pkl')
print("\nModel and preprocessor saved successfully.")