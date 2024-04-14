from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the dataset
data = pd.read_csv('dataset.csv')

# Encode categorical values
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Load the trained model
model = load_model('model.keras')

@app.route('/', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json
    new_input = pd.DataFrame(input_data, index=[0])

    # Encode the input data
    for col in new_input.columns:
        if new_input[col].dtype == 'object':
            new_input[col] = le.transform(new_input[col])

    # Make a prediction
    prediction = model.predict(new_input)

    # Find the index of the maximum probability
    predicted_disease_index = prediction.argmax()
    predicted_disease = data['Disease'].iloc[predicted_disease_index]

    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    app.run(debug=True)
