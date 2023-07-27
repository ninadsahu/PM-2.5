import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from datetime import datetime
MODELS_FOLDER='models'
def days_between(d1,d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)
    
def return_predictions(time, city, date):
    df = pd.read_feather('data.feather')
    where = []
    for n, i in enumerate(list(df["Time Periods"])):
        if i[-5:] != time:
            where.append(n)
    df = df.drop(index=where)
    df = df[df['City'] == city]
    X = np.array(df["PM2.5"])
    cleanedList = [x for x in X if str(x) != 'nan']
    len(cleanedList)
    X = np.array(cleanedList)
    model_to_use=f'{city}_{time}'
    model_path = os.path.join(MODELS_FOLDER, model_to_use)
    model = joblib.load(model_path)
    
    predictions = model.predict(start=X.shape[0], end=X.shape[0]+days_between("2022-12-31",date), dynamic=True)
    final_pred = predictions[days_between("2022-12-31",date)-1]
    return final_pred

    
app = Flask(__name__)
@app.route('/health')
def index():
    return json.dumps({'Health': 'UP'})

@app.route('/api/return_predictions',methods=["GET"])
def return_prediction_get():
    request_data = request.json
    t = request_data['time']
    c = request_data['city']
    d = request_data['date']
    response =  return_predictions(t,c,d)

    response_data = {
        'status': 'success',
        'message': 'Request processed successfully',
        'response': json.dumps(response.tolist())
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')