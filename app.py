
from flask import Flask, request, render_template
import joblib
import pandas as pd


app = Flask(__name__)

model = joblib.load('xgb_cl_prevision_credit.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    d={}
    i=0
    for col in ["gender", "Nb_enf", "Vehicule", "Propriétaire_logement", "Montant_revenu","Montant_credit", "DAYS_BIRTH", "Montant_Rente"]:
        d[col] = int_features[i]
    
    final_features = pd.DataFrame(d, index=[0])
    prediction = ("Ce client est éligible au prêt demandé :) !" if int(model.predict(final_features)) else "Ce client ne semble pas remplir toutes les conditions pour bénéficier du prêt souhaité :(")

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)

