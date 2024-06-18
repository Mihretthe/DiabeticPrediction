from django.shortcuts import render
import pickle
import pandas as pd

def index(request):
    if request.method == 'POST':
        pregnancies = int(request.POST['pregnancies'])
        glucose = int(request.POST['glucose'])
        blood_pressure = int(request.POST['blood_pressure'])
        skin_thickness = int(request.POST['skin_thickness'])
        insulin = int(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        diabetes_pedigree_function = float(request.POST['diabetes_pedigree_function'])
        age = int(request.POST['age'])

        # Load the trained model
        with open('prediction_diabetis_model.pickle', 'rb') as f:
            rf = pickle.load(f)

        # Make a prediction
        prediction = predict_diabetes(
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree_function, age
        )
        prediction_text = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
        return render(request, 'home.html', {'prediction': prediction_text})

    return render(request, 'home.html')

from predict import rf as model

def predict_diabetes( pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    new_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    new_data_df = pd.DataFrame(new_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    prediction = model.predict(new_data_df)
    return prediction[0]