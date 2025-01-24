from flask import Flask, render_template, request
import pandas as pd
import joblib  # For loading the trained model
from datetime import datetime

app = Flask(__name__)

# Load the trained Logistic Regression model
try:
    model = joblib.load('logistic_regression_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Please make sure 'logistic_regression_model.pkl' exists.")
    model = None  # Handle the error as needed

# --- Feature Engineering (from Date) ---
def extract_features_from_date(date_str):
    """
    Extracts features from a date string (YYYY-MM-DD).
    """
    try:
        date = pd.to_datetime(date_str)
        month = date.month
        day_of_year = date.dayofyear
        # Example: Map month to a season (Southern Hemisphere)
        if month in [12, 1, 2]:
            season = 4  # Summer
        elif month in [3, 4, 5]:
            season = 1  # Autumn
        elif month in [6, 7, 8]:
            season = 2  # Winter
        else:
            season = 3  # Spring

        return {
            'Month': month,
            'DayOfYear': day_of_year,
            'Season': season
        }
    except ValueError:
        return None

# --- Prediction Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            date_str = request.form['date']
            features = {
                'MinTemp': float(request.form['MinTemp']),
                'MaxTemp': float(request.form['MaxTemp']),
                'Rainfall': float(request.form['Rainfall']),
                'Evaporation': float(request.form['Evaporation']),
                'Sunshine': float(request.form['Sunshine']),
                'WindGustSpeed': float(request.form['WindGustSpeed']),
                'WindDir9am': float(request.form['WindDir9am']),
                'WindDir3pm': float(request.form['WindDir3pm']),
                'Humidity9am': float(request.form['Humidity9am']),
                'Humidity3pm': float(request.form['Humidity3pm']),
                'Pressure9am': float(request.form['Pressure9am']),
                'Pressure3pm': float(request.form['Pressure3pm']),
                'Cloud9am': float(request.form['Cloud9am']),
                'Cloud3pm': float(request.form['Cloud3pm']),
                'Temp9am': float(request.form['Temp9am']),
                'Temp3pm': float(request.form['Temp3pm']),
                'RainToday': 1 if request.form['RainToday'].lower() == 'yes' else 0,
                'WindGustDir': float(request.form['WindGustDir']),
            }

            # Extract date features
            date_features = extract_features_from_date(date_str)
            if date_features is None:
                return render_template('index.html', error="Invalid date format")

            # Combine all features
            features.update(date_features)

            # Create DataFrame and reorder columns to match the model
            input_data = pd.DataFrame([features])
            input_data = input_data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                                     'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'Humidity9am', 'Humidity3pm',
                                     'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                                     'RainToday', 'Month', 'DayOfYear', 'Season']]

            # Make prediction
            if model is None:
                raise ValueError("Model not loaded.")
            prediction = model.predict(input_data)[0]
            prediction_text = "Yes" if prediction == 1 else "No"

            return render_template('index.html', prediction=prediction_text)
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error="An unexpected error occurred: " + str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
