from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import json
import numpy as np
import math
from ModelPrediction import ModelPrediction


app = Flask(__name__)
CORS(app)

@app.route('/api/test/headers', methods=['POST'])
def headers():
    #Get headers for payload
    headers = ['time_pregnant', 'glucose', 'blood_pressure', 'skin_fold_thick', 'serum_insuling', 'mass_index']

    payload = request.json['data']
    #prediction = model.predict([[np.array(data['exp'])]])
    #output = prediction[0]
    values = [float(i) for i in payload.split(',')]

    input_variables = pd.DataFrame([values],
                                columns=headers,
                                dtype=float,
                                index=['input']) 
    sr = pd.Series([19.5, 16.8, 22.78, 20.124, 18.1002])
    #reurn input_variables.to_jason(orient='records')
    return input_variables.to_json(orient='records')
    
    @app.route('/api/test/headers', methods=['POST'])
    def load():
        ModelPrediction = ModelPrediction()
        ModelPrediction.loadModel('models/isolation_forest.pickle')
        print('leido pikle')
        return "leido"
    @app.route('/api/test/headers', methods=['POST'])
    def predict():
        #{'DateObserved':now, 'H': 2, 'V': 2, 'C': 0.1}

        payload = request.json

        #leo la matriz para poder predecir
        print(payload)
        data = pd.jason_normalize(payload)
        #calculo

        ModelPrediction = ModelPrediction()
        model = modelPrediction.loadModel('models/isolation_forest.pickle')
        dfResultados = ModelPrediction.predict(model, data)

        return Response(dfResultados.to_json(orient='records'), mimetype='application/json')
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug = True)

          