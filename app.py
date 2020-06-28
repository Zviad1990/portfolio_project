from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
import pandas as pd
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time
import xgboost as xgb

# Пробный запуск Flask

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run
model =xgb.Booster()
model.load_model('models/avg_claim.model')

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.route("/predict", methods=['POST'])
def predict():
    json_input = request.json

    # Request logging
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()

    id = json_input['ID']
    df_get = process_input(json_input)
    predictions_gb = pd.DataFrame()

    prediction_gb['predict']=model.predict(df_get)
    result_gb=prediction_gb['predict'][0]
    result = {
        'ID': id,
        'value_ClaimsCount': result_gb
    }

    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datatime} predicted for {duration} msec: {result}\n')

    return jsonify(result)



@app.errorhandler(Exception)
def exceptions(e):
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    error_message = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 current_datatime,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_message)
    return jsonify({'error': 'Internal Server Error'}), 500

@app.route("/")
def index():
    return "API for predict service"

if __name__ == '__main__':
    app.run()
