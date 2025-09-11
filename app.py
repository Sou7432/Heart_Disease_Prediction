from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


with open('soumya.pkl', 'rb') as model_file:
    ml_model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':

        try:
            features =[
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
            input_data = np.array(features).reshape(1,-1)
            result = ml_model.predict(input_data)[0]
            heart_disease = 'Yes' if result == 1 else 'No'
            return render_template('result.html', prediction=heart_disease)
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
if __name__ == "__main__":
    app.run(debug=True)
