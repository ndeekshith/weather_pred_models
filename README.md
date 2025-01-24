# Title: Weather Prediction Project

# Description:
This project is a web-based application that predicts weather conditions using a logistic regression model. Built with Flask, it allows users to input weather features and get predictions on whether it will rain tomorrow.

# Features:

    User-friendly web interface.
    Predicts rainfall using logistic regression.
    Extracts date-based features like month, day of the year, and season.

# Installation:

    Clone the repository.
    Create a virtual environment and activate it.
    Install the dependencies using requirements.txt.
    Ensure the trained model file (logistic_regression_model.pkl) is in the project directory.

# Usage:

    Start the Flask application using python app.py.
    Open http://127.0.0.1:5000/ in your browser.
    Input weather data and get predictions.

## Input Features:

    Weather data like temperatures, humidity, cloud cover, etc.
    Date input to derive additional features.

# Folder Structure:

    app.py: Flask app backend.
    logistic_regression_model.pkl: Trained logistic regression model.
    templates/index.html: Frontend UI.
    model.py : training all the models
    requirements.txt: List of dependencies.

# Contributing:
Open to contributions via issues or pull requests.
