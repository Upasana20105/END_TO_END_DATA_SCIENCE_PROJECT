# END_TO_END_DATA_SCIENCE_PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: UPASANA PRAJAPATI

*INTERN ID*: CT08DF387

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH


*DESCRIPTION OF THE TASK*:

---

# 🏠 House Price Prediction Web App

This project is a web-based application that predicts house prices based on user-provided features. It uses machine learning for prediction and Flask for serving the model through a web interface.

---

## 📁 Project Structure

### 1. `app.py` - 🔌 Flask Backend

* Main entry point of the web application.
* Loads the pre-trained model and preprocessing pipeline (`.pkl` files).
* Defines routes:

  * `/` renders the HTML frontend (`index.html`).
  * `/predict` accepts JSON data, preprocesses it, predicts house price, and returns the result.
* Handles input feature alignment and transformation before prediction.
* Includes error logging using `traceback`.

### 2. `data_processing.py` - 🧠 Model Training & Preprocessing

* Loads the **Kaggle Ames Housing Dataset** (`train.csv`, `test.csv`).
* Performs:

  * Exploratory Data Analysis (EDA),
  * Missing value imputation,
  * Feature scaling and one-hot encoding.
* Trains a `RandomForestRegressor` on log-transformed prices.
* Evaluates the model and saves:

  * `house_price_model.pkl`
  * `house_price_preprocessor.pkl`
  * `numerical_features.pkl`
  * `categorical_features.pkl`

### 3. `index.html` - 🌐 Frontend Form

* A simple and clean web form styled with CSS.
* Allows users to enter house features like:

  * `GrLivArea`, `OverallQual`, `GarageCars`, `MSZoning`, etc.
* Sends the data to Flask backend using JavaScript `fetch()` API via JSON POST request.
* Displays predicted price or errors in a user-friendly manner.

---

## 🚀 How to Run the App

1. **Install dependencies**

   ```bash
   pip install flask pandas numpy scikit-learn seaborn matplotlib joblib
   ```

2. **Ensure you have the following files in your directory:**

   * `train.csv`, `test.csv` from the Kaggle Ames Housing dataset
   * `app.py`, `data_processing.py`, `index.html`

3. **Train the model**

   ```bash
   python data_processing.py
   ```

4. **Run the web app**

   ```bash
   python app.py
   ```

5. **Visit** `http://localhost:5000` in your browser.

---

## 🧠 ML Model Details

* Model: **Random Forest Regressor**
* Target: `SalePrice` (log-transformed)
* Preprocessing:

  * Numeric: Imputation + Scaling
  * Categorical: Imputation + One-Hot Encoding
* Metrics: RMSE and R² score evaluated on validation split

---

### 1. Data Visualizations & Analysis

* **`graph_1.png`**: This image displays the **distribution of `SalePrice`**, the target variable for our prediction model. It is a histogram with a kernel density estimate (KDE) curve, showing the frequency of different house sale prices. This visualization helps understand the target variable's spread and skewness.
* **`graph_2.png`**: This is a **correlation matrix of numerical features**. It visualizes the pairwise correlation between various numerical features in the dataset. Red colors indicate a strong positive correlation, blue colors indicate a strong negative correlation, and lighter shades indicate weaker correlations. This is crucial for feature selection and understanding relationships between variables.

### 2. Application & User Interface

* **`Main_App_page.png`**: This screenshot shows the **main page of the House Price Predictor web application**. It demonstrates the user interface where users can input various house features (e.g., Gross Living Area, Overall Quality, Garage Cars) and get a predicted house price. The predicted price is displayed at the bottom of the form.

### 3. Data Processing & Model Training Logs

The `process_X.jpg` series of images document the data processing and model training steps, typically displayed in a terminal or integrated development environment (IDE).

* **`process_1.jpg`**: Shows the start of the data processing script (`python data_processing.py`). It displays the initial shapes of the training and testing datasets and the `info()` output of the training data, including the number of entries and columns (features) with their data types and non-null counts.
* **`process_2.jpg`**: Continues the `info()` output of the training data, listing more columns and their respective information.
* **`process_3.jpg`**: Completes the `info()` output for the training data, showing the remaining columns and a summary of data types and memory usage.
* **`process_4.jpg`**: Begins displaying the **missing values in the training data**, showing the count of `null` entries for features like `PoolQC`, `MiscFeature`, `Alley`, etc.
* **`process_5.jpg`**: Continues the list of missing values in the training data and starts showing **missing values in the test data**.
* **`process_6.jpg`**: Completes the list of missing values in the test data, providing a comprehensive overview of data sparsity for various features in both datasets.
* **`process_7.jpg`**: Shows the conclusion of the preprocessing steps. It displays the **shapes of the processed training and testing datasets** (`X_train_processed_shape`, `X_test_processed_shape`). Importantly, it also presents the **model's performance metrics on the validation set**:
    * **RMSE (Root Mean Squared Error)** on log-transformed data: `0.1449`
    * **R-squared**: `0.8875`
    * **RMSE on Original Scale**: `29437.65`
    Finally, it confirms that the "Model and preprocessor saved successfully" and indicates that the Flask application is being served.

---

## 🔖 Acknowledgements

* [Kaggle Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* Flask Web Framework
* scikit-learn for ML pipeline

---

## 📸 Output
---
![Image](https://github.com/user-attachments/assets/b6dfb972-a635-48f9-8a92-112e6414858f)
![Image](https://github.com/user-attachments/assets/242dada4-1f9e-4a5f-af3f-8912cc68ffdf)
![Image](https://github.com/user-attachments/assets/facb8ae8-1113-4a20-bc68-ed051bfdb3e7)
![Image](https://github.com/user-attachments/assets/92c8940a-b50d-4fd9-b93c-e9768c7c353d)
![Image](https://github.com/user-attachments/assets/b0683cd3-9756-42ac-9c83-ad4c11dc34b0)
![Image](https://github.com/user-attachments/assets/d24f40bd-0b9c-4292-9621-09383b850899)
![Image](https://github.com/user-attachments/assets/ca4b8886-aab2-4c88-8774-9c349c7894ba)
![Image](https://github.com/user-attachments/assets/089097db-4e83-416f-a921-6283a91481a1)
![Image](https://github.com/user-attachments/assets/b1df7caf-6788-48fe-b469-17f6eb011be6)
![Image](https://github.com/user-attachments/assets/b56cbeb0-4797-4256-a808-e960488817d7)
![Image](https://github.com/user-attachments/assets/79376786-c52c-4e8d-ba10-cd7c9cd33557)

