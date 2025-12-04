from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Expecting features as list: {"features": [..]}
    features = data.get('features', [])
    pred = model.predict([features])
    return jsonify({'prediction': pred[0]})

if __name__ == '__main__':
    app.run(debug=True)
