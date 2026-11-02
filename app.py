"""Web app: draw a digit in the browser and let the model guess it.

Usage:
    python app.py        then open http://127.0.0.1:5000
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import load_model, predict_image  # noqa: E402
from preprocess import decode_data_url  # noqa: E402

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify(error="No image received."), 400
    try:
        result = predict_image(decode_data_url(image_data), get_model())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
