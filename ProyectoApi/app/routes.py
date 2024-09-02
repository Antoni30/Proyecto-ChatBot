from flask import Blueprint, request, jsonify
from .model import predict

main = Blueprint('main', __name__)

@main.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    features = data['query']
    prediction = predict(features)
    return jsonify({'respuesta': prediction})

@main.route('/hello', methods=['GET'])
def hello():
    return "Hola, Mundo"