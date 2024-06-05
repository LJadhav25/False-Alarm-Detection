import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/home')
def home_page():
    return render_template('AlarmDetection.html')


@app.route('/training', methods=['GET'])
def train():
    alarm_cases = pd.read_excel(r"C:\Users\lokes\Downloads\Historical Alarm Cases.xlsx")
    x = alarm_cases.iloc[:, 1:7]
    y = alarm_cases['Spuriosity Index(0/1)']
    log = LogisticRegression()
    log.fit(x, y)
    joblib.dump(log, 'train.pkl')
    # alarm_cases_json = alarm_cases.to_json(orient='records')
    return "Model trained successfully"


@app.route('/submit', methods=['POST'])
def false_alarm():
    training_file = joblib.load('train.pkl')
    false = "False Alarm, No Danger"
    true = "True Alarm, Danger"
    if request.method == 'POST':
        ambient_temp = request.form['at']
        calibration = request.form['cd']
        usd = request.form['usd']
        humidity = request.form['hm']
        contentment = request.form['hots']
        detected = request.form['detected']
        test_data = [ambient_temp, calibration, usd, humidity, contentment, detected]
        my_data_array = np.array(test_data)
        test_array = my_data_array.reshape(1, 6)
        test = pd.DataFrame(test_array,
                            columns=['Ambient Temperature( deg C)', 'Calibration(days)',
                                     'Unwanted substance deposition(0/1)', 'Humidity(%)',
                                     'H2S Content(ppm)', 'detected by(% of sensors)'])
        y_predict = training_file.predict(test)
        if y_predict == 1:
            return render_template('contact.html', false_alarm=false)
        else:
            return render_template('contact.html', true_alarm=true)


@app.route('/predict', methods=['POST'])
def predict():
    training_file = joblib.load('train.pkl')
    data = request.get_json()
    c1 = data['Ambient Temperature']
    c2 = data['Calibration']
    c3 = data['Unwanted substance deposition']
    c4 = data['Humidity']
    c5 = data['H2S Content']
    c6 = data['detected by']
    test_data = [c1, c2, c3, c4, c5, c6]
    my_data_array = np.array(test_data)
    test_array = my_data_array.reshape(1, 6)
    test = pd.DataFrame(test_array,
                        columns=['Ambient Temperature( deg C)', 'Calibration(days)',
                                 'Unwanted substance deposition(0/1)', 'Humidity(%)',
                                 'H2S Content(ppm)', 'detected by(% of sensors)'])
    y_predict = training_file.predict(test)
    if y_predict == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger"


app.run(port=5000)
