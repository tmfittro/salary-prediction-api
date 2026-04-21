from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        print("INCOMING:", data)

        age = data.get("age")
        gender = data.get("gender")
        country = data.get("country")
        highest_deg = data.get("highest_deg")
        coding_exp = data.get("coding_exp")
        title = data.get("title")

        if None in [age, gender, country, highest_deg, coding_exp, title]:
            return {"error": "Missing one or more required fields"}, 400

        input_data = [[age, gender, country, highest_deg, coding_exp, title]]
        prediction = model.predict(input_data)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}, 500