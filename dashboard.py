# Display a dashboard in streamlit in order to
# diplay and explain client's scoring for credit

# Import required librairies
import streamlit as st
import streamlit.components.v1 as components
from urllib.request import urlopen
import json
import datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap

# Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "OC - P7 - Scoring Client", layout="wide")
# API configuration
# local :
#API_url = "http://127.0.0.1:5000/"
# online :
API_url = "http://bl0ws.pythonanywhere.com/"
# Initialize javascript for shap plots
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

# Get all the clients data through an API
json_url_all = urlopen(API_url + "data")
API_data_all = json.loads(json_url_all.read())
data_all = pd.DataFrame(API_data_all)

# Create the list of clients
client_list = data_all["SK_ID_CURR"].tolist()

# Create the list of columns
columns = list(data_all.iloc[:,1:21].columns)

# Create the reference data (mean, median, mode)
Z = data_all[columns]
data_ref = pd.DataFrame(index=Z.columns)
data_ref["mean"] = Z.mean()
data_ref["median"] = Z.median()
data_ref["mode"] = Z.mode().iloc[0, :]
data_ref = data_ref.transpose()

# In the sidebar allow to select a client in the list
st.sidebar.header("Paramètres")
client_id = st.sidebar.selectbox("Identification du client",
                                 client_list)

# Store the index in the DataFrame for this client
client_index = data_all[data_all["SK_ID_CURR"] == client_id].index

# Prepare the data for the comparison plot by scaling it
data_plot = data_all[columns]
col_one = []
col_std = []
for col in data_plot.columns:
    if data_plot[col].min() >= 0 and data_plot[col].max() <= 1:
        col_one.append(col)
    else:
        col_std.append(col)
scale_min_max = ColumnTransformer(
    transformers=[
        ("std", MinMaxScaler(), col_std),
    ],
    remainder="passthrough",
)
data_plot_std = scale_min_max.fit_transform(data_plot)
data_plot_final = pd.DataFrame(data_plot_std, columns=data_plot.columns)
data_client = data_plot_final.loc[client_index, :]

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
columns_selected = st.sidebar.multiselect("Informations du client à afficher",
                                 columns, default)

# Once the client and columns are selected, run the process
# display a message while processing...
with st.spinner("Traitement en cours..."):
    # Get the data for the selected client and the prediction from an API
    json_url_client = urlopen(API_url + "data/client/" + str(client_id))
    API_data_client = json.loads(json_url_client.read())
    df = pd.DataFrame(API_data_client)
    
    # Store the columns names to use them in the shap plots
    client_data = df.iloc[0:1,0:20]
    features_analysis = client_data.columns
    
    # store the data we want to explain in the shap plots
    data_explain = np.asarray(client_data)
    shap_values = df.iloc[1,0:20].values
    expected_value = df["expected"][0]
    
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
    
    # in an expander, display the client's data and comparison with average
    with st.expander("Ouvrir pour afficher l'analyse détaillée"):
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
        
    st.subheader("Caractéristiques du client")
    
    # Display a plot that compares the current client within all the clients
    # Initialize the figure
    f, ax = plt.subplots(figsize=(7, 5))
    # Set the style for average values markers
    meanpointprops = dict(markeredgecolor="black", markersize=8,
                              markerfacecolor="green", markeredgewidth=0.66)
    # Build the boxplots for each feature
    sns.boxplot(
            data=data_plot_final[columns_selected],
            orient="h",
            whis=3,
            palette="muted",
            linewidth=0.7,
            width=0.6,
            showfliers=False,
            showmeans=True,
            meanline=False,
            meanprops=meanpointprops)
    # Add in a point to show current client
    sns.stripplot(
            data=data_client,
            orient="h",
            size=8,
            palette="blend:firebrick,firebrick",
            marker="D",
            edgecolor="black",
            linewidth=0.66)
    # Remove ticks labels for x
    ax.set_xticklabels([])
    # Manage y labels style
    ax.set_yticklabels(columns_selected,
            fontdict={"fontsize": "medium",
                "fontstyle": "italic",
                "verticalalignment": "center",
                "horizontalalignment": "right"})
    # Remove axes lines
    sns.despine(trim=True, left=True, bottom=True, top=True)
    # Removes ticks for x and y
    plt.tick_params(left=False, bottom=False)
    # Add separation lines for y values
    lines = [ax.axhline(y, color="grey", linestyle="solid", linewidth=0.7)
                            for y in np.arange(0.5, len(columns_selected), 1)]
    # Proxy artists to add a legend
    average = mlines.Line2D([], [], color="green", marker="^",
                            linestyle="None", markeredgecolor="black",
                            markeredgewidth=0.66, markersize=8, label="moyenne")
    current = mlines.Line2D([], [], color="firebrick", marker="D",
                            linestyle="None", markeredgecolor="black",
                            markeredgewidth=0.66, markersize=8, label="client courant")
    ax.legend(handles=[average, current], bbox_to_anchor=(1, 1), fontsize="small")
    # Display the plot
    st.pyplot(f)
    
    # in an expander, display the client's data and comparison with average
    with st.expander("Ouvrir pour afficher les données détaillées"):
        temp_df = pd.concat([client_data, data_ref])
        new_df = temp_df.transpose()
        new_df.columns = ["Client (" + str(client_id) + ")", "Moyenne",
                          "Médiane", "Mode"]
        st.table(new_df.loc[columns_selected,:])

# Display a success message in the sidebar once the process is completed
with st.sidebar:
    end = datetime.datetime.now()
    text_success = "Last successful run : " + str(end.strftime("%Y-%m-%d %H:%M:%S"))
    st.success(text_success)