import warnings
import numpy as np
import pickle
import os
import traceback
import logging
from flask import Flask, request, jsonify, render_template, redirect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Specific Sklearn Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Load Pretrained Models and Scalers
telecom_model = None
banking_model = None
telecom_scaler = None
banking_scaler = None

try:
    logger.info("Loading models and scalers...")
    
    with open('models/telecom_model.pkl', 'rb') as f:
        telecom_model = pickle.load(f)
        logger.info("Telecom model loaded successfully")

    with open('models/banking_model.pkl', 'rb') as f:
        banking_model = pickle.load(f)
        logger.info("Banking model loaded successfully")

    with open('models/telecom_scaler.pkl', 'rb') as f:
        telecom_scaler = pickle.load(f)
        logger.info("Telecom scaler loaded successfully")

    with open('models/banking_scaler.pkl', 'rb') as f:
        banking_scaler = pickle.load(f)
        logger.info("Banking scaler loaded successfully")

except Exception as e:
    error_trace = traceback.format_exc()
    logger.error(f"Error loading models or scalers: {e}")
    logger.error(f"Traceback: {error_trace}")
    # Don't exit here - allow the app to start anyway so we can diagnose

# Function to Parse Telecom Customer Form Data
def parse_telecom_form(form_data):
    try:
        logger.info(f"Parsing telecom form data: {form_data}")
        
        tenure = int(form_data.get('tenure', 0))
        monthly_charges = float(form_data.get('monthly_charges', 0.0))
        total_charges = float(form_data.get('total_charges', 0.0))
        paperless_billing = int(form_data.get('paperless_billing', 0))
        senior_citizen = int(form_data.get('senior_citizen', 0))
        streaming_tv = int(form_data.get('streaming_tv', 0))
        streaming_movies = int(form_data.get('streaming_movies', 0))
        multiple_lines = int(form_data.get('multiple_lines', 0))
        phone_service = int(form_data.get('phone_service', 0))
        device_protection = int(form_data.get('device_protection', 0))
        online_backup = int(form_data.get('online_backup', 0))
        partner = int(form_data.get('partner', 0))
        dependents = int(form_data.get('dependents', 0))
        tech_support = int(form_data.get('tech_support', 0))
        online_security = int(form_data.get('online_security', 0))
        gender = form_data.get('gender', '')
        
        logger.info("Processing categorical variables...")
        
        # One-hot Encoding Categorical Variables
        contract = form_data.get('contract', '')
        internet_service = form_data.get('internet_service', '')
        payment_method = form_data.get('payment_method', '')

        contract_encoded = [1 if contract == "Month-to-month" else 0,
                            1 if contract == "One year" else 0,
                            1 if contract == "Two year" else 0]

        internet_service_encoded = [1 if internet_service == "Fiber optic" else 0,
                                    1 if internet_service == "DSL" else 0,
                                    1 if internet_service == "No" else 0]

        payment_method_encoded = [1 if payment_method == "Electronic check" else 0,
                                  1 if payment_method == "Mailed check" else 0,
                                  1 if payment_method == "Bank transfer (automatic)" else 0,
                                  1 if payment_method == "Credit card (automatic)" else 0]

        gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]

        # Create Features Array
        features = np.array([paperless_billing, senior_citizen, streaming_tv, streaming_movies,
                             multiple_lines, phone_service, device_protection, online_backup,
                             partner, dependents, tech_support, online_security,
                             monthly_charges, total_charges, tenure] +
                            contract_encoded + internet_service_encoded + payment_method_encoded + gender_encoded).reshape(1, -1)
        
        logger.info(f"Generated feature array with shape: {features.shape}")
        return features

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        raise ValueError(f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error processing telecom form data: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Error processing form data: {e}")

# Function to Parse Bank Customer Form Data
def parse_banking_form(form_data):
    try:
        logger.info(f"Parsing banking form data: {form_data}")
        
        credit_score = int(form_data.get('credit_score', 0))
        age = int(form_data.get('age', 0))
        tenure = int(form_data.get('tenure', 0))
        balance = float(form_data.get('balance', 0.0))
        num_of_products = int(form_data.get('num_of_products', 0))
        has_cr_card = int(form_data.get('has_cr_card', 0))
        is_active_member = int(form_data.get('is_active_member', 0))
        estimated_salary = float(form_data.get('estimated_salary', 0.0))
        satisfaction_score = int(form_data.get('satisfaction_score', 0))
        point_earned = int(form_data.get('point_earned', 0))
        gender = form_data.get('gender', '')
        card_type = form_data.get('card_type', '')
        
        logger.info("Processing categorical variables...")

        # One-hot Encoding Categorical Variables
        gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
        card_type_encoded = [1 if card_type == "DIAMOND" else 0,
                             1 if card_type == "GOLD" else 0,
                             1 if card_type == "SILVER" else 0,
                             1 if card_type == "PLATINUM" else 0]

        # Create Feature Array
        features = np.array([credit_score, age, tenure, balance, num_of_products,
                             has_cr_card, is_active_member, estimated_salary,
                             satisfaction_score, point_earned] +
                            gender_encoded + card_type_encoded).reshape(1, -1)
        
        logger.info(f"Generated feature array with shape: {features.shape}")
        return features

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        raise ValueError(f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error processing banking form data: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Error processing form data: {e}")

# Test endpoint to verify models are loaded
@app.route('/test-models', methods=['GET'])
def test_models():
    try:
        logger.info("Testing model loading status")
        
        # Test if models are loaded
        telecom_model_loaded = telecom_model is not None
        banking_model_loaded = banking_model is not None
        telecom_scaler_loaded = telecom_scaler is not None
        banking_scaler_loaded = banking_scaler is not None
        
        # Check model shapes/sizes if possible
        telecom_model_info = str(type(telecom_model)) if telecom_model else "Not loaded"
        banking_model_info = str(type(banking_model)) if banking_model else "Not loaded"
        
        return jsonify({
            'telecom_model_loaded': telecom_model_loaded,
            'banking_model_loaded': banking_model_loaded,
            'telecom_scaler_loaded': telecom_scaler_loaded,
            'banking_scaler_loaded': banking_scaler_loaded,
            'telecom_model_type': telecom_model_info,
            'banking_model_type': banking_model_info
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in test-models route: {e}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500

# Echo endpoint to verify form data is correctly received
@app.route('/api/echo-data', methods=['POST'])
def echo_data():
    try:
        logger.info("Received request to echo-data endpoint")
        data = request.get_json()
        logger.info(f"Received data: {data}")
        return jsonify({
            'status': 'success',
            'data_received': data
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in echo-data route: {e}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500

# Predict Bank Churn
@app.route('/api/bank-churn-prediction', methods=['POST'])
def predict_banking():
    try:
        logger.info("Request received at /api/bank-churn-prediction")
        
        # Get and log the form data
        form_data = request.get_json()
        logger.info(f"Form data received: {form_data}")
        
        # Parse form data and log
        logger.info("Parsing form data...")
        user_data = parse_banking_form(form_data)
        logger.info(f"Parsed user data shape: {user_data.shape}")
        
        # Scale data and log
        logger.info("Transforming data with scaler...")
        if banking_scaler is None:
            raise ValueError("Banking scaler is not loaded")
        user_data_scaled = banking_scaler.transform(user_data)
        logger.info(f"Scaled data shape: {user_data_scaled.shape}")
        
        # Make prediction and log
        logger.info("Making prediction with model...")
        if banking_model is None:
            raise ValueError("Banking model is not loaded")
        prediction = banking_model.predict(user_data_scaled)
        logger.info(f"Prediction: {prediction}")
        
        # Return result
        prediction_result = "Churned" if prediction[0] == 1 else "Not Churned"
        logger.info(f"Returning prediction: {prediction_result}")
        return jsonify({'prediction': prediction_result})
    
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"ValueError in predict_banking: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Exception in predict_banking: {error_msg}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': error_msg,
            'traceback': error_trace
        }), 500

# Predict Telecom Churn
@app.route('/api/telecom-churn-prediction', methods=['POST'])
def predict_telecom():
    try:
        logger.info("Request received at /api/telecom-churn-prediction")
        
        # Get and log the form data
        form_data = request.get_json()
        logger.info(f"Form data received: {form_data}")
        
        # Parse form data and log
        logger.info("Parsing form data...")
        user_data = parse_telecom_form(form_data)
        logger.info(f"Parsed user data shape: {user_data.shape}")
        
        # Scale data and log
        logger.info("Transforming data with scaler...")
        if telecom_scaler is None:
            raise ValueError("Telecom scaler is not loaded")
        user_data_scaled = telecom_scaler.transform(user_data)
        logger.info(f"Scaled data shape: {user_data_scaled.shape}")
        
        # Make prediction and log
        logger.info("Making prediction with model...")
        if telecom_model is None:
            raise ValueError("Telecom model is not loaded")
        prediction = telecom_model.predict(user_data_scaled)
        logger.info(f"Prediction: {prediction}")
        
        # Return result
        prediction_result = "Churned" if prediction[0] == 1 else "Not Churned"
        logger.info(f"Returning prediction: {prediction_result}")
        return jsonify({'prediction': prediction_result})
    
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"ValueError in predict_telecom: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Exception in predict_telecom: {error_msg}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': error_msg,
            'traceback': error_trace
        }), 500
    
# Simple navigation routes - both return to index
@app.route('/bank-prediction')
@app.route('/telecom-prediction')
def return_to_index():
    return redirect('/')    

# About Me Page
@app.route('/aboutme')
@app.route('/AboutMe.html')
def aboutme():
    try:
        return render_template('AboutMe.html')
    except Exception as e:
        logger.error(f"Error loading AboutMe page: {str(e)}")
        return f"Error loading AboutMe page: {str(e)}", 500

# Home Page 
@app.route('/')
@app.route('/index.html')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return f"Error loading home page: {str(e)}", 500

# Fix Heroku Deployment Port Issue
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Bind to dynamic port
    app.run(host='0.0.0.0', port=port, debug=False)