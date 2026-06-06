from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

# Flask app
app = Flask(__name__)

# Absolute path of current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(BASE_DIR, "soumya.pkl")

with open(model_path, "rb") as model_file:
    ml_model = pickle.load(model_file)

# Feature names (must match training data)
feature_names = [
    'age',
    'sex',
    'Chest pain type',
    'BP',
    'Cholesterol',
    'FBS over 120',
    'EKG results',
    'Max HR',
    'Exercise angina',
    'ST depression',
    'Slope of ST',
    'Number of vessels fluro',
    'Thallium'
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

        input_df = pd.DataFrame([features], columns=feature_names)

        prediction = ml_model.predict(input_df)[0]

        heart_disease = "Yes" if prediction == 1 else "No"

        return render_template(
            'result.html',
            prediction=heart_disease
        )

    except Exception as e:
        return render_template(
            'result.html',
            prediction=f"Error: {str(e)}"
        )

# For local testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
