from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('LinearRegression.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_Data.csv') 


@app.route('/', methods=['GET', 'POST'])
def index():
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    car_models.insert(0, 'Select Car')
    return render_template('cardsmp.html', car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():

    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, year, driven, fuel_type]).reshape(1, 4)))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()
