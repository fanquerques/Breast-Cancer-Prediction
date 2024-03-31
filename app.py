# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:27:11 2024

@author: Fan Yang, Dejing Chen
"""

from flask import Flask, request, render_template
import joblib
import numpy as np
import json

app = Flask(__name__)

model = joblib.load(r"C:\Users\maily\Desktop\GenAi_BPC\breast_cancer_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    n_features = [float(i) for i in request.form.values()]
    features = [np.array(n_features)]
    prediction = model.predict(features)
    return render_template("result.html", prediction=prediction[0])

@app.route("/model_info")
def model_info():
    # Load model parameters and performance metrics
    with open('model_metrics.json') as f:
        model_metrics = json.load(f)
    return render_template("model_info.html", model_metrics=model_metrics)

if __name__=="__main__":
    app.run(debug=True, port=8000)
