from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from model import AdvancedMLP
import numpy as np

app = Flask(__name__)

# Model parameters
input_size = 4
hidden_size = 64
output_size = 3

# Initialize and load model
try:
    model = AdvancedMLP(input_size, hidden_size, output_size)
    model_path = 'model/advanced_iris_model.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    
# Feature normalization parameters (use the same as training)
feature_means = [5.843333, 3.057333, 3.758000, 1.199333]
feature_stds = [0.828066, 0.435866, 1.765298, 0.762238]

def normalize_features(features):
    return [(f - m) / s for f, m, s in zip(features, feature_means, feature_stds)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]
        
        # Input validation
        if any(f <= 0 for f in features):
            return jsonify({'error': 'All measurements must be positive numbers'})
            
        # Normalize features
        normalized_features = normalize_features(features)
        
        # Make prediction
        predicted_class = infer(normalized_features)
        
        return jsonify({
            'prediction': predicted_class,
            'features': features
        })
        
    except ValueError:
        return jsonify({'error': 'Please enter valid numerical values'})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def infer(features):
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        prediction = model(features_tensor)
        predicted_class = torch.argmax(prediction).item()
        return ['setosa', 'versicolor', 'virginica'][predicted_class]

if __name__ == '__main__':
    app.run(debug=True)
