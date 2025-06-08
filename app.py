from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import traceback # For better error logging

app = Flask(__name__)

# Load the trained model and preprocessor
try:
    model = joblib.load('house_price_model.pkl')
    preprocessor = joblib.load('house_price_preprocessor.pkl')
    # Get feature names after one-hot encoding for consistent input structure
    # This part requires careful handling as preprocessor.get_feature_names_out()
    # is available for ColumnTransformer
    
    # Store the original feature names from the training data, excluding 'Id' and 'SalePrice'
    # This assumes the order and presence of features will be consistent with new data
    global_numerical_features = joblib.load('numerical_features.pkl') # Save these during preprocessing
    global_categorical_features = joblib.load('categorical_features.pkl') # Save these during preprocessing

except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    # Handle the error appropriately, e.g., exit or raise an exception
    exit()

# Helper function to preprocess incoming data
def preprocess_input(data):
    # Convert incoming dictionary to a DataFrame
    input_df = pd.DataFrame([data])
    
    # Ensure all expected columns are present, fill missing with NaN if not
    # This is crucial for consistent input to the preprocessor
    expected_columns = global_numerical_features + global_categorical_features
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan # Or appropriate default value

    # Reorder columns to match the order used during training (important for ColumnTransformer)
    input_df = input_df[expected_columns]

    # Preprocess the input data using the loaded preprocessor
    processed_input = preprocessor.transform(input_df)
    return processed_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded.'}), 500

    try:
        json_ = request.json
        # Convert dictionary to DataFrame for consistent preprocessing
        # It's important that the keys in the JSON match your feature names
        
        # Example expected input structure (replace with your actual features):
        # {
        #     "GrLivArea": 1710,
        #     "OverallQual": 7,
        #     "GarageCars": 2,
        #     "GarageArea": 548,
        #     "TotalBsmtSF": 856,
        #     "1stFlrSF": 856,
        #     "FullBath": 2,
        #     "TotRmsAbvGrd": 8,
        #     "YearBuilt": 2003,
        #     "MSSubClass": 60,
        #     "MSZoning": "RL",
        #     "LotFrontage": 65.0,
        #     "LotArea": 8450,
        #     "Street": "Pave",
        #     "Alley": "NA", # Example of how missing values could be sent
        #     "LotShape": "Reg",
        #     "LandContour": "Lvl",
        #     "Utilities": "AllPub",
        #     "BldgType": "1Fam",
        #     "HouseStyle": "2Story",
        #     "RoofStyle": "Gable",
        #     "Exterior1st": "VinylSd",
        #     "Exterior2nd": "VinylSd",
        #     "ExterQual": "Gd",
        #     "ExterCond": "TA",
        #     "Foundation": "PConc",
        #     "BsmtQual": "Gd",
        #     "BsmtCond": "TA",
        #     "HeatingQC": "Ex",
        #     "CentralAir": "Y",
        #     "KitchenQual": "Gd",
        #     "Functional": "Typ",
        #     "FireplaceQu": "NA",
        #     "GarageType": "Attchd",
        #     "GarageFinish": "RFn",
        #     "PavedDrive": "Y",
        #     "SaleType": "WD",
        #     "SaleCondition": "Normal"
        # }
        
        processed_data = preprocess_input(json_)
        prediction_log = model.predict(processed_data)
        prediction_original = np.expm1(prediction_log)[0] # Take the first element as predict returns an array

        return jsonify({'predicted_house_price': round(prediction_original, 2)})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        app.logger.error(traceback.format_exc()) # Log the full traceback
        return jsonify({'error': 'Prediction failed. Please check your input data.', 'details': str(e)}), 400

if __name__ == '__main__':
    # Save the feature lists during the preprocessing step in the Jupyter Notebook/script
    # For a robust application, these should be consistently generated or manually defined
    # and saved alongside the model and preprocessor.
    # For this example, we'll quickly create them here as well.
    # In a real project, this would be part of your data pipeline.
    
    # Reload original data to get feature names, assuming 'Id' and 'SalePrice' are dropped
    original_train_df = pd.read_csv('train.csv')
    original_test_df = pd.read_csv('test.csv')
    combined_features_df = pd.concat([original_train_df.drop(['Id', 'SalePrice'], axis=1), 
                                      original_test_df.drop('Id', axis=1)], ignore_index=True)

    global_numerical_features = combined_features_df.select_dtypes(include=np.number).columns.tolist()
    global_categorical_features = combined_features_df.select_dtypes(include='object').columns.tolist()

    joblib.dump(global_numerical_features, 'numerical_features.pkl')
    joblib.dump(global_categorical_features, 'categorical_features.pkl')

    app.run(debug=True, host='0.0.0.0', port=5000) # Run on all interfaces for potential external access