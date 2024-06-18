import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('diabetes.csv')

# Split the data into features (X) and target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Function to accept new data and make a prediction
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    new_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    new_data_df = pd.DataFrame(new_data, columns=X.columns)
    prediction = rf.predict(new_data_df)
    return prediction[0]

# Example usage
new_data_point = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
prediction = predict_diabetes(*new_data_point)
print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")

import pickle
with open('prediction_diabetis_model.pickle','wb') as f:
    pickle.dump(rf,f)