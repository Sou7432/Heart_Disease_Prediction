from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='../templates')

# Load model safely (absolute path for Vercel)
model_path = os.path.join(os.path.dirname(__file__), '..', 'soumya.pkl')
with open(model_path, 'rb') as model_file:
    ml_model = pickle.load(model_file)

# Feature names (IMPORTANT: must match training time)
feature_names = [
    'age', 'sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['Chest pain type']),
            float(request.form['BP']),
            float(request.form['Cholesterol']),
            float(request.form['FBS over 120']),
            float(request.form['EKG results']),
            float(request.form['Max HR']),
            float(request.form['Exercise angina']),
            float(request.form['ST depression']),
            float(request.form['Slope of ST']),
            float(request.form['Number of vessels fluro']),
            float(request.form['Thallium'])
        ]

        # Fix feature name warning using DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)

        result = ml_model.predict(input_df)[0]
        heart_disease = 'Yes' if result == 1 else 'No'

        return render_template('result.html', prediction=heart_disease)

    except Exception as e:
        return f"Error: {str(e)}"


# ✅ REQUIRED for Vercel serverless
def handler(request):
    return app(request.environ, lambda *args: None)