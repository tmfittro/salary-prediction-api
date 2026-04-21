from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔥 CRITICAL FIX: read JSON input
        data = request.get_json()

        print("INCOMING DATA:", data)

        # extract values
        age = data.get("age")
        gender = data.get("gender")
        country = data.get("country")
        highest_deg = data.get("highest_deg")
        coding_exp = data.get("coding_exp")
        title = data.get("title")

        # check for missing fields
        if None in [age, gender, country, highest_deg, coding_exp, title]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        # prepare input for model
        input_data = [[age, gender, country, highest_deg, coding_exp, title]]

        # prediction
        prediction = model.predict(input_data)[0]

        # return result
        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)