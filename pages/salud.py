# salud .py
'''

'''
# librerias
import os
import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards

import base64
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from millify import millify
import pygwalker as pgw
# from pandasai import SmartDatalake  # para multiples dataframes
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse

from datetime import datetime
from funciones import load_data_csv
from dotenv import load_dotenv

from funciones import load_data_csv

# configuration
st.set_page_config(
    page_title="Maceda",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
API_KEY = st.secrets['OPENAI_API_KEY']  # os.environ['OPENAI_API_KEY']
openai_api_key = API_KEY

df = load_data_csv(
    "https://data.vgclab.cl/public_data/dataset_licencias_sample.csv")

# --------------------------
# METRICAS
# --------------------------
confirmadas = df[df["ESTADO"] == "CONFIRMADA"]
confirmadas = confirmadas["ESTADO"].value_counts().tolist()
confirmadas = sum(confirmadas)


# --------------------------
# MAIN
# --------------------------
tabPanel, tabTable, tabIA, tabBleau, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Análisis",  "Información"])

with tabPanel:
    st.write("Panel")
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Confirmadas", millify(confirmadas, precision=0))
    col2.metric("Regiones", 2)
    col3.metric("Tipo de Actores", 2)
    col4.metric(label="Ataques", value=2)
    col5.metric(label="Protestas violentas", value=2)

    style_metric_cards()

with tabTable:
    st.dataframe(df, height=500)

with tabIA:
    st.write("IA-EDA")


with tabBleau:
    report = pgw.walk(df, return_html=True)
    components.html(report, height=1000, scrolling=True)

with tabInfo:
    st.write("Información")
    st.write("")
