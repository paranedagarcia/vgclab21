'''
Detenciones.
Fecha: 2024-01-12
Autor: Patricio Araneda
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
    page_title="Eficiencia de justicia",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

API_KEY = st.secrets['OPENAI_API_KEY']  # os.environ['OPENAI_API_KEY']
openai_api_key = API_KEY


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            date = row[0]
            year = date.split('/')[2]
            print(year)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# Chile Regions, counties and provinces

comunas_file = "https://data.vgclab.cl/public_data/comunas.csv"
comunas = pd.read_csv(comunas_file, header=0, sep=";")
regiones = comunas["REGION"].unique().tolist()
provincias = comunas["PROVINCIA"].unique().tolist()

datos = "https://data.vgclab.cl/public_data/dataset_detenciones.csv"
# datos = "data/dataset_detenciones.csv"
try:
    df = load_data_csv(datos)
# df = pd.read_csv("data/dataset_detenciones.csv", header=0, sep=",")
# Convert 'FECHA' to datetime type
except:
    st.error("Error al cargar los datos")
    st.stop()

df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

# Extract year and create 'YEAR' column
df['YEAR'] = df['FECHA'].dt.year

delitos = df["DELITO"].unique().tolist()


delito_selection = st.sidebar.multiselect("Delitos:", delitos, [
                                          'Rb_Violencia_o_Intimidación', 'Rb_Sorpresa', 'Rb_Fuerza', 'Rb_Vehìculo', 'Hurtos'])

# medios involucrados

# filter delitos
# selected_unidad = st.sidebar.selectbox(
#     "Seleccione agrupación", ["Ninguna", "Comuna", "Provincia", "Región"])

df = df[df["DELITO"].isin(delito_selection)]

# regiones = df["REGION"].unique().tolist()
# regiones = [x for x in regiones if str(x) != 'nan']
# region = st.sidebar.multiselect(
#     "Seleccione una o mas regiones", regiones, default=regiones)

# if region is not None:
#     df = df[df["REGION"].isin(region)]
# else:
#     df = df

st.subheader("Eficiencia de la Justicia")
# st.subheader("Detenciones " + selected_unidad)

# METRICAS

# st.write("Total de detenciones: ", df.shape[0])
detenciones_totales = df.shape[0]
delitos_totales = df["DELITO"].nunique()

tab1, tab2, tab3, tab4 = st.tabs(["Panel", "Tabla", "IA-EDA", "Análisis"])
with tab1:

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Total de detenciones", millify(
        detenciones_totales, precision=2))
    col2.metric("Tipo de delitos", delitos_totales)
    col3.metric("", None)
    col4.metric("", None)

    style_metric_cards()

    # if selected_unidad == "Comuna":
    #     df = df.groupby(
    #         ["COMUNA", "DELITO", "YEAR"]).agg({'CASOS': sum})

    # elif selected_unidad == "Región":
    #     df = df.groupby(
    #         ["REGION", "DELITO", "YEAR"]).agg({'CASOS': sum})

    # else:
    #     pass

    #
    det = df.groupby(["COMUNA", "DELITO", "YEAR"]).agg({'CASOS': sum})

    # Carga de datos (asegúrate de ajustar la ruta del archivo según tu entorno)
    # df = pd.read_csv('ruta/a/tu/dataset_detenciones.csv')

    # Calcula el total de casos para cada tipo de delito
    crime_counts = df.groupby('DELITO')['CASOS'].sum().reset_index()

    # Calcula el total de casos para el cálculo del porcentaje
    total_cases = crime_counts['CASOS'].sum()

    # Calcula el porcentaje de cada tipo de delito
    crime_counts['Porcentaje'] = (crime_counts['CASOS'] / total_cases) * 100

    # Ordena los delitos por número de casos para obtener los delitos más comunes
    crime_counts_sorted = crime_counts.sort_values('CASOS', ascending=False)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig = px.pie(df, values=df["DELITO"].value_counts().tolist(), names=df["DELITO"].unique().tolist(),
                     title='Porcentaje de casos por delito en todas las regiones')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("2")

    # Mostrar el gráfico en Streamlit
    fig = px.bar(df, x="YEAR", y="CASOS",
                 color="DELITO", title="Casos por año")
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.dataframe(df, height=500)

with tab3:  # EDA IA
    st.subheader("Análisis exploratorio con Inteligencia Artificial")
    with st.expander("Información importante"):
        st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el dataset de MACEDA. Ingrese su consulta la que pudiera ser respondida por el modelo en forma de texto o una imagen gráfica.")
        st.write(
            "Por ejemplo, puede preguntar: ¿Cuántos eventos de tipo 'X' ocurrieron en la región 'Y' en el año '2018'?")
        st.warning(
            "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")
    st.write("")

    llm = OpenAI(client=OpenAI, streaming=True,
                 api_token=API_KEY, temperature=0.5)

    prompt = st.text_area("Ingrese su consulta:")

    if st.button("Generar respuesta"):
        if prompt:
            with st.spinner("Generando respuesta... por favor espere."):
                sdf = SmartDataframe(df, config={"llm": llm})
                st.write(sdf.chat(prompt))
        else:
            st.write("Por favor ingrese una consulta.")


with tab4:
    report = pgw.walk(df, return_html=True)
    components.html(report, height=1000, scrolling=True)
