from flask import Flask, request, jsonify
from app.models.infer import predict
import os

app = Flask(__name__)

MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "models/best_svm.joblib")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    if isinstance(data, dict):
        result = predict(MODEL_PATH, data)
        return jsonify(result)
    elif isinstance(data, list):
        results = [predict(MODEL_PATH, sample) for sample in data]
        return jsonify(results)
    else:
        return jsonify({"error": "Input must be a dict or list of dicts"}), 400


# For local development, run: python app.py
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
