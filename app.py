import joblib
import pandas as pd
import numpy
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Set up the Flask app
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the models
models = {
    'gradient_boosting': joblib.load(os.path.join(BASE_DIR, 'model/gradient_boosting_model.pkl')),
    'logistic_regression': joblib.load(os.path.join(BASE_DIR, 'model/logistic_regression_model.pkl')),
    'random_forest': joblib.load(os.path.join(BASE_DIR, 'model/random_forest_model.pkl')),
    'decision_tree': joblib.load(os.path.join(BASE_DIR, 'model/decision_tree_model.pkl'))
}

# Input data structure
class InputData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Home route
@app.route('/')
def home():
    return """
Welcome to the prediction API. Please send a POST request to /predict with the required data.
"""

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract input data and model name
    try:
        model_name = data['model_name']
        if model_name not in models:
            return jsonify({"error": "Model not found"}), 400

        # Remove the model_name from input data for prediction
        input_data = {key: value for key, value in data.items() if key != 'model_name'}
        input_df = pd.DataFrame([input_data])

        # Get the model
        model = models[model_name]
        
        # Predict and calculate probability
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]
        
        # Prepare message with prediction and probability
        probab_perc = f'{(probability[0] * 100):.2f}'
        message = probab_perc if prediction[0] == 1 else probab_perc
        
        # Return the result
        return jsonify({
            'model': model_name,
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'message': message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2000)
