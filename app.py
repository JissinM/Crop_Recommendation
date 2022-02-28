import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
@flask_app.route("/analysis")
def analysis():
    return render_template("crop.html")


model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("home.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("home.html", prediction_text = "The Suggested Crop for Given Climatic Condition is : {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)