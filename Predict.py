import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/predict')
def predict():
    model = pickle.load(open('model.pkl','rb'))
    data = request.args.get('data')
    arr = data.split(",")
    prediction = model.predict([np.array(arr)])
    output = prediction[0]
    return jsonify(output)

@app.route("/build")
def build():
    url = 'https://docs.google.com/spreadsheet/ccc?key=1T3iyoPQdI1FlJrhr-l-YjK-NlED_JfyjT7lGezZN1rY&output=csv'
    X=pd.read_csv(url,",")
    y=X['GlucoseLevel']
    X = X.drop(['Date', 'Time', 'GlucoseLevel','Prediction'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    reg = LinearRegression().fit(X_train, y_train)
    pickle.dump(reg, open('model.pkl','wb'))
    
    html = "<H1>Model Updated Succesfully...</H1>"
    html = html + "Train Score: " + str(reg.score(X_train, y_train)) + "<br/>"
    html = html + "Train Coef: " + str(reg.coef_)  + "<br/>"
    html = html + "Train Intercept: " + str(reg.intercept_)  + "<br/>"
    html = html + "Test Score: " + str(reg.predict(X_test))  + "<br/>"
    html = html + "Test Score: " + str(reg.score(X_test, y_test))  + "<br/>"
    return html