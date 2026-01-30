import joblib
import numpy as np
import sklearn

model = joblib.load('./model/insurance-ml.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

age = int(input("Ingrese edad : "))
age_sc = sc_x.transform(np.array([[age]]))

prediction = model.predict(age_sc)
prediction_sc = sc_y.inverse_transform(prediction)
print(f'El precio de seguro para una persona con  {age} años es de {prediction_sc[0][0]:.2f} USD')