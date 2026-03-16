from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.route("/")
def home():
    return "ML API is running successfully!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Please provide features in JSON"}), 400

        features = np.array(data["features"])

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
