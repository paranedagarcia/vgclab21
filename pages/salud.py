# salud .py
'''
Analisis de  salud mental  basado en licencias médicas otorgadas por la Superintendencia de Seguridad Social de Chile.
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
import seaborn as sns

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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

load_dotenv()
API_KEY = st.secrets['OPENAI_API_KEY']  # os.environ['OPENAI_API_KEY']
openai_api_key = API_KEY


filename = "https://data.vgclab.cl/public_data/dataset_licencias_sample.csv"

columnas = ['FECHA', 'LICENCIA', 'FORMULARIO', 'TRABAJADOR', 'EMISION', 'INICIO', 'DIAS', 'CODIGO', 'TIPO', 'PROFESIONAL', 'ENTIDAD', 'CAJA', 'ESTADO', 'DESDE', 'HASTA',
            'DIASAUT', 'SUBSIDIO', 'CODAUT', 'AUTORIZACION', 'CODPREV', 'RECHAZO', 'SUBCIE10', 'GRUPOCIE10', 'CIE10', 'LIQUIDO', 'SUSESO']
dtipos = {
    'FECHA': np.int16,
    'LICENCIA': str,
    'TRABAJADOR': str,
    'PROFESIONAL': str,
    'DIAS': np.int16,
    'DIASAUT': np.int16,
    'CODIGO': str,
    'CODAUT': str,
    'CODPREV': str,
}


df = pd.read_csv(filename, encoding='utf-8', sep=",",
                 na_values='NA',
                 usecols=columnas,
                 dtype=dtipos
                 )
df['LIQUIDO'] = df['LIQUIDO'].fillna(0).astype(int)
df['SUSESO'] = df['SUSESO'].fillna(0).astype(int)
df['EMISION'] = pd.to_datetime(df['EMISION'])
df['DESDE'] = pd.to_datetime(df['DESDE'])
df['HASTA'] = pd.to_datetime(df['HASTA'])


# filtro por  años
# years = df['FECHA'].unique().tolist()
# years.insert(0, "Todos")
# anual = st.sidebar.selectbox("Seleccione un año", years)

# if anual is not 'Todos':
#     df = df[df["FECHA"] == anual]
# else:
#     pass

st.sidebar.subheader("Licencias médicas ")
# Filtro de rango de fechas
min_date = df['EMISION'].min()
max_date = df['EMISION'].max()
date_start = st.sidebar.date_input(
    'Emisión desde', value=pd.to_datetime(min_date))
date_end = st.sidebar.date_input(
    'Emisión hasta', value=pd.to_datetime(max_date))

df = df[
    (df['EMISION'] >= pd.to_datetime(date_start)) &
    (df['EMISION'] <= pd.to_datetime(date_end))]


# Filtro por SUBCIE10
# cie10 = df['CIE10'].unique().tolist()
# cie10.insert(0, "Todos")
# cie10_to_filter = st.sidebar.multiselect(
#     'CIE10', cie10, default="Todos")
# if cie10_to_filter is not 'Todos':
#     df = df[df["CIE10"] == cie10_to_filter]
# else:
#     pass


# --------------------------
# METRICAS
# --------------------------

total_records = df.shape[0]
total_licencias = df["LICENCIA"].nunique()
total_trabajadores = df["TRABAJADOR"].nunique()
total_profesional = df["PROFESIONAL"].nunique()
total_rechazadas = df[df["ESTADO"] == "RECHAZADA"].shape[0]
total_autorizadas = df[df["ESTADO"] == "CONFIRMADA"].shape[0]

# --------------------------
# MAIN
# --------------------------
st.subheader(f"Licencias por salud mental")

tabPanel, tabTable, tabIA, tabBleau, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Análisis",  "Información"])

with tabPanel:
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Licencias", millify(total_licencias, precision=2))
    col2.metric("Rechazadas", millify(total_rechazadas, precision=2))
    col3.metric("Autorizadas", millify(total_autorizadas, precision=2))
    col4.metric("Solicitantes", millify(total_trabajadores, precision=2))
    col5.metric("Profesionales", millify(total_profesional, precision=2))

    style_metric_cards()

    # -------------------------------------------
    coll, colr = st.columns(2, gap="medium")

    with coll:
        fig = px.pie(df, names='ESTADO',
                     title='Estado de las licencias')
        st.plotly_chart(fig, use_container_width=True)

    with colr:
        fig = px.pie(df, names='TIPO', title='Tipo de licencia')
        st.plotly_chart(fig, use_container_width=True)

    # Convert 'EMISION' to datetime format and extract year-month
    # df['EMISION'] = pd.to_datetime(df['EMISION'])
    # df['YearMonth'] = df['EMISION'].dt.to_period('M')
    # df_licencia = df.groupby('YearMonth')['LICENCIA'].count().reset_index()

    # fig_licencia = px.line(df_licencia, x='YearMonth',
    #                        y='LICENCIA', title='Licencias por Mes', markers=True)
    # fig_licencia.update_layout(
    #     xaxis_title='Fecha', yaxis_title='Número de Licencias')
    # st.plotly_chart(fig_licencia, use_container_width=True)

    # ----------------------------------------------
    # Create a line plot for 'LIQUIDO' and 'SUSESO' based on 'FECHA'

    df_line = df.groupby('FECHA')[['LIQUIDO', 'SUSESO']].sum().reset_index()
    fig_line = px.line(df_line, x='FECHA', y=[
                       'LIQUIDO', 'SUSESO'], title='Tendencias de pagos de licencia', markers=True)
    fig_line.update_layout(title={'xanchor': 'center', 'x': 0.5})
    st.plotly_chart(fig_line, use_container_width=True)

    # ----------------------------------------------
    # Create a line plot for 'LIQUIDO' and 'SUSESO' based on 'EMISION'

    df_line_emission = df.groupby(
        'EMISION')[['LIQUIDO', 'SUSESO']].sum().reset_index()
    fig_line_emission = px.line(df_line_emission, x='EMISION', y=[
                                'LIQUIDO', 'SUSESO'],
                                title='Tendencia de emisión de licencias y pagos')
    fig_line_emission.update_layout(showlegend=True, xaxis_title=None, yaxis_title=None,
                                    title={'xanchor': 'center', 'x': 0.5},
                                    legend=dict(title=None,
                                                yanchor='top', xanchor='center', x=0.5))
    st.plotly_chart(fig_line_emission, use_container_width=True)

    # ----------------------------------------------
    # Create a bar plot for 'LIQUIDO' and 'SUSESO' based on 'FECHA'

    df_bar = df.groupby('FECHA')[['LIQUIDO', 'SUSESO']].sum().reset_index()
    fig_bar = px.bar(df_bar, x='FECHA', y=[
                     'LIQUIDO', 'SUSESO'], title='Tendencias de pagos en Años')
    fig_bar.update_layout(yaxis_title=None, xaxis_title='FECHA',
                          title={'xanchor': 'center', 'x': 0.5},
                          legend=dict(title=None,
                                      yanchor='top', xanchor='center', x=0.5))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------
    # Configuraciones de visualización
    st.write("Tendencias a lo largo del tiempo para la emisión de licencias médicas.")
    st.divider()
    sns.set(style="whitegrid")

    # 1. Tendencias a lo largo del tiempo para 'EMISION'
    df['Año'] = df['EMISION'].dt.year
    df['Mes'] = df['EMISION'].dt.month

    # Agrupar por año y mes para contar el número de licencias emitidas
    licencias_por_mes = df.groupby(
        ['Año', 'Mes']).size().reset_index(name='Cantidad')

    # Visualización
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=licencias_por_mes, x='Año', y='Cantidad',
                 estimator='sum', ci=None, marker='o')
    plt.title('Tendencia de Licencias Médicas Emitidas por Año', fontsize=18)
    plt.xlabel('Año', fontsize=14)
    plt.ylabel('Cantidad de licencias emitidas', fontsize=14)
    plt.xticks(licencias_por_mes['Año'].unique())
    st.pyplot(plt)

    # ----------------------------------------------
    # 2. Distribución del estado de las licencias médicas ('ESTADO')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='ESTADO',
                  order=df['ESTADO'].value_counts().index)
    plt.title('Distribución de Estados de Licencias Médicas')
    plt.xlabel('Cantidad')
    plt.ylabel('Estado')
    st.pyplot(plt)

    # ----------------------------------------------
    # 3. Análisis de la columna 'AUTORIZACION'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='AUTORIZACION',
                  order=df['AUTORIZACION'].value_counts().index)
    plt.title('Distribución de Autorizaciones de Licencias Médicas')
    plt.xlabel('Cantidad')
    plt.ylabel('Autorización')
    st.pyplot(plt)

    st.divider()
    # ----------------------------------------------
    coll, colr = st.columns(2, gap="medium")
    with coll:
        st.write("La visualización corregida muestra la tendencia mensual de licencias médicas emitidas en el período. A través de esta gráfica, podemos observar cómo la cantidad de licencias varía mes a mes a lo largo del período de estudio.")
    with colr:
        st.write("Podría haber patrones estacionales evidentes, donde ciertos meses muestran variación en el número de licencias emitidas. Estos patrones podrían estar relacionados con factores estacionales como enfermedades comunes en ciertas épocas del año.")
    st.write("")

    # Corregir el formato de 'AñoMes' para visualización, convirtiéndolo a cadena
    if 'AñoMes' not in df.columns:
        df['AñoMes'] = df['Año'].astype(
            str) + '-' + df['Mes'].astype(str)
    # data['AñoMes'] = data['AñoMes'].astype(str)

    # Agrupar nuevamente por 'AñoMes' corregido para contar el número de licencias emitidas
    licencias_por_anomes_corregido = df.groupby(
        'AñoMes').size().reset_index(name='Cantidad')

    # Visualización corregida
    plt.figure(figsize=(18, 6))
    sns.lineplot(data=licencias_por_anomes_corregido,
                 x='AñoMes', y='Cantidad', marker='o')
    plt.title('Tendencia Mensual de Licencias Médicas Emitidas', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Cantidad de Licencias Médicas Emitidas', fontsize=14)
    plt.xticks(rotation=90)  # Rotar las etiquetas para mejor visualización
    plt.tight_layout()
    st.pyplot(plt)

    st.divider()
    # ----------------------------------------------
    # variables estacionales
    licencias_por_mes_promedio = df.groupby(
        'Mes').size().reset_index(name='Cantidad')
    licencias_por_mes_promedio['CantidadPromedio'] = licencias_por_mes_promedio['Cantidad'] / len(
        df['Año'].unique())

    # Visualización
    plt.figure(figsize=(12, 6))
    sns.barplot(data=licencias_por_mes_promedio, x='Mes',
                y='CantidadPromedio', palette='coolwarm')
    plt.title('Promedio de Licencias Médicas Emitidas por Mes', fontsize=16)
    plt.xlabel('Mes', fontsize=14)
    plt.ylabel('Promedio de Licencias Médicas Emitidas', fontsize=14)
    plt.xticks(range(0, 12), ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])
    plt.tight_layout()
    st.pyplot(plt)

    # ----------------------------------------------
    # codigos CIE10
    # Identificar los 10 códigos de diagnóstico (CIE10) más comunes
    st.subheader("CIE10")
    st.write("Patrones específicos por condición de salud: Ciertos códigos de diagnóstico muestran picos claros en meses específicos, lo que puede indicar la estacionalidad de estas condiciones. Por ejemplo, si los códigos relacionados con enfermedades respiratorias aumentan en los meses de invierno, esto sugeriría una influencia estacional.")
    st.write("")
    top_cie10 = df['CIE10'].value_counts().nlargest(10).index.tolist()

    # Filtrar los datos para incluir solo los registros con los códigos de diagnóstico más comunes
    data_top_cie10 = df[df['CIE10'].isin(top_cie10)]

    # Agrupar por mes y código de diagnóstico para contar el número de licencias emitidas
    licencias_por_mes_cie10 = data_top_cie10.groupby(
        ['Mes', 'CIE10']).size().reset_index(name='Cantidad')

    # Visualización de la distribución mensual de los 10 códigos de diagnóstico más comunes
    plt.figure(figsize=(18, 10))
    sns.lineplot(data=licencias_por_mes_cie10, x='Mes',
                 y='Cantidad', hue='CIE10', marker='o', palette='tab10')
    plt.title(
        'Distribución Mensual de Licencias Médicas por Top 10 Códigos de Diagnóstico (CIE10)', fontsize=20)
    plt.xlabel('Mes', fontsize=16)
    plt.ylabel('Cantidad de Licencias Médicas Emitidas', fontsize=16)
    plt.xticks(range(1, 13), ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'], fontsize=14)
    plt.legend(title='Código de Diagnóstico (CIE10)')
    plt.tight_layout()
    st.pyplot(plt)

    coll, colr = st.columns(2, gap="medium")
    coll.write("Diferencias en los patrones estacionales: No todas las condiciones de salud siguen el mismo patrón estacional, lo cual es esperado. Algunas condiciones pueden ser consistentemente altas a lo largo del año, mientras que otras muestran variabilidad mensual significativa.")
    colr.write("Importancia del contexto clínico y social: Para interpretar estos patrones correctamente, es crucial considerar el contexto clínico de cada código de diagnóstico (CIE10) y posibles factores sociales o ambientales que puedan influir en estas tendencias.")


# -------------------------------------------
with tabTable:
    st.dataframe(df, height=500)

# -------------------------------------------
with tabIA:
    st.subheader("Análisis exploratorio con Inteligencia Artificial")
    with st.expander("Información importante"):
        st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el dataset de MACEDA. Ingrese su consulta la que pudiera ser respondida por el modelo en forma de texto o una imagen gráfica.")
        st.write(
            "Por ejemplo, puede preguntar: ¿Cuántos eventos de tipo 'X' ocurrieron en la región 'Y' en el año '2018'?")
        st.warning(
            "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")
    st.write("")
    user_path = os.getcwd()
    # llm = OpenAI(api_token=API_KEY)
    llm = OpenAI(client=OpenAI, streaming=True,
                 api_token=API_KEY, temperature=0.5)

    prompt = st.text_area("Ingrese su consulta:")

    if st.button("Generar respuesta"):
        if prompt:
            with st.spinner("Generando respuesta... por favor espere."):
                llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
                # query = SmartDataframe(df, config={"llm": llm})
                query = Agent(df, config={"llm": llm,
                                          "save_charts": False,
                                          # "save_charts_path": user_path,
                                          "open-charts": True,
                                          "verbose": True,
                                          "response_parser": StreamlitResponse
                                          })

                response = query.chat(prompt)

                if isinstance(response, str) and response.endswith("png"):
                    st.image(response)
                else:
                    st.write(response)
        else:
            st.write("Por favor ingrese una consulta.")

# -------------------------------------------
with tabBleau:
    report = pgw.walk(df, return_html=True)
    components.html(report, height=1000, scrolling=True)

# -------------------------------------------
with tabInfo:
    st.write("Información")
    st.write("Analisis de  salud mental  basado en licencias médicas otorgadas por la Superintendencia de Seguridad Social de Chile.")
