# Library Imports
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from spacy.lang.en.stop_words import STOP_WORDS

# Load trained model
model = joblib.load('c2_Sentimentanalysis_Model_pipeline.pkl')
stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)

# Define Predict Function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]

    # Assuming you have a pipeline defined as pipeline_svc
    predictions = model.predict(new_review)[0]

    if predictions == 0:
        return render_template('index.html', prediction_text='Negative')
    else:
        return render_template('index.html', prediction_text='Positive')

if __name__ == "__main__":
    app.run(debug=True)
