import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the dataset
data = pd.read_csv('dataset.csv')

# Encode categorical values
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Assuming 'model' is your trained model loaded from a file
model = load_model('model.keras')

# Prepare a new input sample (e.g., the first row of your dataset)
new_input = data.drop('Disease', axis=1).iloc[0].values.reshape(1, -1)

# Make a prediction
prediction = model.predict(new_input)


print(f"Predicted Disease: {prediction}")

import numpy as np



# Find the index of the maximum probability
predicted_disease_index = np.argmax(prediction)

predicted_disease_index = 15  # Example predicted disease index
predicted_disease = data['Disease'].iloc[predicted_disease_index]
print(f"Predicted Disease: {predicted_disease}")


