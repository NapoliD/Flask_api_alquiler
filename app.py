import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cols = ['barrio_n','ambientes','m2']

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    data_unseen = pd.DataFrame([final_features], columns = cols)

    prediction = model.predict(data=data_unseen)

    prediction = int(prediction[0])
    '''prediction = model.predict(final_features)'''

    output = int(np.where(prediction>0,prediction,0))

    return render_template('index1.html', prediction_text='Alquiler estimado $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)