from flask import Flask, request, render_template
# import pickle, sklearn

import pandas as pd

import numpy as np
import joblib
model = joblib.load("lr_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    d = {}
    i = 0
    val = [int (x) for x in request.form.values()]
    for col in ['CODE_GENDER', 'CNT_CHILDREN', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'class_amt_tot', 'class_amt_credit', 'class_amt_ANNUITY', 'class_age']:		
        d[col] = val[i]
        i += 1
    inputs = pd.DataFrame(d, index=[0])
    # bd = pd.read_json('donnees_entrees_form.json')
    # pd.concat([bd, inputs],0).reset_index(drop=True).to_json("donnees_entrees_form.json")

    #inputs.to_csv('donnees_entrees_formulaire.csv')
    resultat = model.predict(inputs)
    reponse  = "Crédit accordé :) !" if resultat == 1 else "Crédit refusé :("
    return render_template('index.html', prediction_text='Prédiction : {}'.format(reponse))



if __name__ == "__main__":
    app.run(debug=True)
    