# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:12:54 2022

@author: bruno
"""

import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
import shap
import os


app = flask.Flask(__name__)
app.config["DEBUG"] = True

#path = "C:/Users/bruno/kDrive/Work/OpenClassrooms/P7/"
path = os.path.dirname(os.path.realpath(__file__))

@app.route('/', methods=['GET'])
def home():
    return """
<h1>OC - P7 - API</h1>
<p>Description de ce qu'on fait ici.</p>

"""

# getting our trained model from a file we created earlier
model = pickle.load(open(path + "/model.pkl","rb"))
# getting the data
X = pd.read_csv(path + "/data_sample.csv")
data_ref = pd.read_csv(path + "/data_ref.csv")

# defining a route to get clients IDs
@app.route("/data/ID", methods=["GET"])
def get_ids():
    ids = {}
    ids["client_id"] = X["SK_ID_CURR"].tolist()
    return jsonify(ids)

# defining a route to get columns names
@app.route("/data/columns", methods=["GET"])
def get_columns():
    columns = {}
    columns["col"] = list(X.iloc[:,1:21].columns)
    return jsonify(columns)

# defining a route to get columns names
@app.route("/data/ref", methods=["GET"])
def get_ref():
    ref = data_ref.to_dict("list")
    return jsonify(ref)

# defining a route to get clients data
@app.route("/data/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # filter the data thanks to the id from the request
    df_sample = X[X["SK_ID_CURR"] == int(client_id)]
    feature_array = np.asarray(df_sample.iloc[0,1:21])

    df_sample["prediction"] = model.predict([feature_array]).tolist()[0]
    df_sample['proba_1'] = model.predict_proba([feature_array])[:,1].tolist()[0]
    
    explainer = shap.KernelExplainer(model.predict_proba, X.iloc[:,1:21]) 
    shap_values = explainer.shap_values(feature_array, l1_reg="aic")
    
    df_sample["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1]) + [0,0,explainer.expected_value[1]]
    new_line2 = [99999] + list(shap_values[0]) + [0,0,explainer.expected_value[0]]
    df_sample.loc[1] = new_line
    df_sample.loc[2] = new_line2
    
    # create the dictionary to be sent
    sample = df_sample.to_dict("list")
    #returning sample and prediction objects as json
    return jsonify(sample)

if __name__ == '__main__':
    app.run(host='0.0.0.0')