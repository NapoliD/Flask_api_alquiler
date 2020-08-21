import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('XGB_Alquiler.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = pd.DataFrame(final_features)
    final_features.columns = ['barrio_cat', 'ambientes', 'm2']
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Alquiler estimado $ {}'.format(int(output)))
'''
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    data = pd.DataFrame.from_dict(data)

    prediction = model.predict(data)

    output = prediction[0]
    return jsonify(output)
'''
if __name__ == "__main__":
    app.run(debug=True)