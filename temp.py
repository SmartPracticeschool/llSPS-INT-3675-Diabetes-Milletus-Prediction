

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('decision.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('abcd-Copy2.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if(output==1):
        pred="diabetic"
    else:
        pred="not diabetic"
    return render_template('abcd-Copy2.html', prediction_text='{}'.format(pred))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)




if __name__ == "_main_":
    app.run(debug=True)
