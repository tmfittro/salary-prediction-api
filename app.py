from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        age = data.get("age")
        gender = data.get("gender")
        country = data.get("country")
        highest_deg = data.get("highest_deg")
        coding_exp = data.get("coding_exp")
        title = data.get("title")

        if None in [age, gender, country, highest_deg, coding_exp, title]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        input_data = [[
            int(age),
            int(gender),
            int(country),
            int(highest_deg),
            int(coding_exp),
            int(title)
        ]]

        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)