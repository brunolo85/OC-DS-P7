# Display a dashboard in streamlit in order to
# diplay and explain client's scoring for credit

# Required librairies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from urllib.request import urlopen
import json
import datetime
import numpy as np
import shap

# streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "OC - P7 - Scoring Client", layout="wide")
# API configuration
API_url = "http://127.0.0.1:5000/"
#API_url = "http://bl0ws.pythonanywhere.com/"
#username = 'Bl0wS'
#token = 'a22bdbd9f871b5a2e2533f35560bf1baa7f269fd'

# shap
shap.initjs()

# Functions
# Display shap force plot
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Dashboard title
st.write("""
# OC - P7 - Scoring Client
Présentation du scoring client et interprétation
""")

# Get the list of clients through an API
json_url_ID = urlopen(API_url + "data/ID")
API_data_ID = json.loads(json_url_ID.read())
df_client = pd.DataFrame(API_data_ID)
client_list = df_client["client_id"].tolist()

# In the sidebar allow to select a client in the list
st.sidebar.header("Paramètres")
client_id = st.sidebar.selectbox("Identification du client",
                                 client_list)

# Get the list of columns through an API so the user can filter
json_url_col = urlopen(API_url + "data/columns")
API_data_col = json.loads(json_url_col.read())
df_col = pd.DataFrame(API_data_col)
columns = df_col["col"].tolist()

# manually define the default columns names
default = [
    "CODE_GENDER",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "PREV_NAME_CONTRACT_STATUS_Approved_MEAN",
    "PREV_CODE_REJECT_REASON_SCOFR_MEAN",
    "PREV_CODE_REJECT_REASON_XAP_MEAN",
]

# In the sidebar allow to select several columns in the list
columns_selected = st.sidebar.multiselect("Informations à afficher",
                                 columns, default)

# Store the reference values for all clients and all columns
json_url_ref = urlopen(API_url + "data/ref")
API_data_ref = json.loads(json_url_ref.read())
data_ref = pd.DataFrame(API_data_ref)

# Once the client and columns are selected, run the process
# display a message while processing...
with st.spinner("Please wait while processing..."):
    # Get the data for the selected client and the prediction from an API
    json_url_client = urlopen(API_url + "data/client/" + str(client_id))
    API_data_client = json.loads(json_url_client.read())
    df = pd.DataFrame(API_data_client)
    
    # Store the columns names to use them in the shap plots
    client_data = df.iloc[0:1,0:20]
    features_analysis = client_data.columns
    
    col1, col2, col3 = st.columns(3)
    if df["proba_1"][0]<0.45:
        with col1:
            st.success("Risque faible")
    elif df["proba_1"][0]>0.55:
        with col3:
            st.error("Risque élevé")
    else:
        with col2:
            st.warning("Risque modéré")
    
    # Display the client's scoring
    st.slider("", min_value=0,
              max_value=100, value=int(round(df["proba_1"][0],2)*100),
                  disabled=True)
    # in an expander, display the client's data and comparison with average
    with st.expander("Caractéristiques du client"):
        temp_df = pd.concat([client_data, data_ref])
        new_df = temp_df.transpose()
        new_df.columns = [client_id, "Moyenne", "Médiane", "Mode"]
        st.write(new_df.loc[columns_selected,:])

    # store the data we want to explain in the shap plots
    data_explain = np.asarray(client_data)
    shap_values = df.iloc[1,0:20].values
    expected_value = df["expected"][0]

    # Explain the scoring thanks to shap plots
    st.subheader('Interprétation du scoring')
    # display a shap force plot
    fig_force = shap.force_plot(
        expected_value,
        shap_values,
        data_explain,
        feature_names=features_analysis,
    ) 
    st_shap(fig_force)
    
    # display a shap waterfall plot
    fig_water = shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values,
        feature_names=features_analysis,
        max_display=10,
    )
    st.pyplot(fig_water)
    
    # display a shap decision plot
    fig_decision = shap.decision_plot(
        expected_value, 
        shap_values, 
        features_analysis)
    st.pyplot(fig_decision)
# Display a success message in the sidebar once the process is completed
with st.sidebar:
    end = datetime.datetime.now()
    text_success = "Last successful run : " + str(end.strftime("%Y-%m-%d %H:%M:%S"))
    st.success(text_success)