import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data.csv')

# Fill missing values with the previous value
data = data.ffill()

# Encode categorical variables
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Print the classes
print(le.classes_)


# Split the data into features and target variable
X = data.drop('Disease', axis=1)
y = data['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') 