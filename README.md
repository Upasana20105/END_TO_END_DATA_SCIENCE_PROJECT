# END_TO_END_DATA_SCIENCE_PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: UPASANA PRAJAPATI

*INTERN ID*: CT08DF387

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH


*DESCRIPTION OF THE TASK*:

Here’s a well-structured **GitHub README description** summarizing all the uploaded files and how they work together in your **House Price Prediction Web App using Machine Learning and Flask** project:

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

## 🔖 Acknowledgements

* [Kaggle Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* Flask Web Framework
* scikit-learn for ML pipeline

---

## 📸 Screenshots
Let me know if you want the same README in Markdown (`README.md`) format for direct GitHub upload!

