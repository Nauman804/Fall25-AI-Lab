from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your updated 6-feature model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Pick exactly the 6 features you selected earlier
    Passengerid = float(request.form["Passengerid"])
    Age = float(request.form["Age"])
    Fare = float(request.form["Fare"])
    Sex = float(request.form["Sex"])
    sibsp = float(request.form["sibsp"])
    Parch = float(request.form["Parch"])

    # Prepare features for prediction
    features = np.array([[Passengerid, Age, Fare, Sex, sibsp, Parch]])

    prediction = model.predict(features)[0]

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
