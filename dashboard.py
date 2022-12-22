import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import flask
import pickle
import os
import numpy as np
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
path = os.path.dirname(__file__)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

st.write("""
# OC - P7 - Scoring Client
Présentation du scoring client et interprétation
""")

X = pd.read_csv(path + '\data_sample.csv')
client_list = X["SK_ID_CURR"]

st.sidebar.header("Identification du client")
client_id = st.sidebar.selectbox("Choisir l'identifiant dans la liste", client_list)

data_input = X[X["SK_ID_CURR"] == client_id]
df = X.iloc[:,1:21]
features_analysis = df.columns

st.subheader("Caractéristiques moyennes des clients")
# Afficher un tableau avec les caractéristiques moyennes des clients

st.subheader("Caractéristiques du client sélectionné")
st.write(data_input)

#getting our trained model from a file we created earlier
model = pickle.load(open(path + "\model.pkl","rb"))

data_input = np.asarray(data_input.iloc[:,1:21])
df = np.asarray(df)

#defining a route for only post requests
@app.route('/predict', methods=['POST'])
def predict(data_input):
    scoring = model.predict(data_input)[0]
    proba = model.predict_proba(data_input)[0]
    if scoring == 0:
        decision = "Favorable (" + str(round(proba[0]*100, 2)) + "%)"
    else:
        decision = "Défavorable (" + str(round(proba[1]*100, 2)) + "%)"
    return decision

st.sidebar.header("Scoring client")
with st.sidebar:
    st.write(predict(data_input))
    #st.write(round(predict(data_input)[1]*100,2) & "% favorable")
    #st.write(1-round(predict(data_input)[1]*100,2) & "% défavorable")

st.subheader('Interprétation du scoring')
shap.initjs()
explainer = shap.KernelExplainer(model.predict_proba, df) 
shap_values = explainer.shap_values(data_input, l1_reg="aic")
fig = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    data_input[0, :],
    feature_names=features_analysis,
) 
st_shap(fig)

shap.initjs()
fig_water = shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value[1],
    shap_values[1][0],
    feature_names=features_analysis,
    max_display=10,
)
#st_shap(fig_water)
st.pyplot(fig_water)