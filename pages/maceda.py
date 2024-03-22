'''
Conflictos mapuches en Chile basadod en datos de MACEDA (Mapuche Chilean State Conflict Event Database).
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
# revisar si se puede usar pandas-profiling
# import pandas_profiling
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

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

load_dotenv()

API_KEY = st.secrets['OPENAI_API_KEY']  # os.environ['OPENAI_API_KEY']
openai_api_key = API_KEY

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


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


local_css("style/style.css")

# procesos
# carga de datos
df = load_data_csv("https://data.vgclab.cl/public_data/dataset_maceda.csv")

# df = pd.read_csv(datos, sep=",", noindex=True, low_memory=False)
# sacar columnas
df = df.drop(columns=["id_evento", "fecha_reportada"])
df = df.rename(columns={'ubicacion_tipo': 'ubicacion',
               'evento_tipo_maceda': 'tipo de evento',
                        'evento_especifico': 'detalle de evento',
                        'actor_tipo_1': 'tipo de actor',
                        'actor_especifico_1': 'actor principal',
                        'actor_especifico_1_armas': 'armas',
                        'actor_tipo_2': 'actor afectado',
                        'actor_mapuche': 'actor mapuche',
                        'mapuche_identificado': 'mapuche identificado',
                        'mesn': 'mes'})

df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
# cambiar fecha al inicio
fecha = df.pop("fecha")
df.insert(0, "fecha", fecha)

# crear dataframe estatico, independiente de selecciones del usuario
serie = df.groupby(['fecha', 'tipo de evento']
                   ).size().reset_index(name='cuenta')
# medios = df.sort_values(by="medio").medio.unique()
# medios.insert(0, "Todos")

# sidebar
years = df["fecha"].dt.year.unique().tolist()
years.insert(0, "Todos")
anual = st.sidebar.selectbox("Seleccione un año", years)

if anual is not 'Todos':
    df = df[df["fecha"].dt.year == anual]
else:
    pass

tipos_eventos = df["tipo de evento"].unique().tolist()
tipo_evento = st.sidebar.multiselect(
    "Seleccione un tipo de evento", tipos_eventos, default=tipos_eventos)

if tipo_evento is not None:
    df = df[df["tipo de evento"].isin(tipo_evento)]
else:
    df = df


regiones = df["region"].unique().tolist()
regiones = [x for x in regiones if str(x) != 'nan']
region = st.sidebar.multiselect(
    "Seleccione una o mas regiones", regiones, default=regiones)

if region is not None:
    df = df[df["region"].isin(region)]
else:
    df = df

# metricas
eventos_totales = df["tipo de evento"].value_counts().tolist()
eventos_totales = sum(eventos_totales)

regiones_totales = df["region"].unique().tolist()
regiones_totales = len(regiones_totales)

actores_totales = df["actor principal"].unique().tolist()
actores_totales = len(actores_totales)

# grafica tipos de eventos (tipo)
eventos = df["tipo de evento"].unique().tolist()
eventos_pie = go.Figure(data=[go.Pie(
    labels=eventos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

# actores involucrados
df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')
df_actor1.plot(kind='barh', x='actor principal', y='cuenta',
               title='actor principal', figsize=(15, 8))

# grafica regiones (linea)
regiones = df["region"].unique().tolist()
regiones_pie = go.Figure(data=[go.Pie(
    labels=regiones, values=df["region"].value_counts().tolist(), hole=.3)])

# grafica eventos
tipos = df["tipo de evento"].unique().tolist()
tipos_bar = go.Figure(data=[go.Pie(
    labels=tipos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

eventos_tiempo = px.line(
    df, x=df["fecha"], y=df["tipo de evento"], color=df["tipo de evento"])

# --------------------------
# MAIN
# --------------------------
st.subheader("Conflictos Mapuches en Chile - MACEDA")

tabPanel, tabTable, tabIA, tabBleau, tabProfile, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Análisis", "Perfil", "Información"])

with tabPanel:  # graficos

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Eventos", millify(eventos_totales, precision=2))
    col2.metric("Regiones", regiones_totales)
    col3.metric("Tipo de Actores", actores_totales)
    col4.metric(label="", value=20)
    col5.metric("", None)

    style_metric_cards()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        # st.subheader("Tipo de evento")
        # st.plotly_chart(eventos_pie, use_container_width=True, title="Eventos en el tiempo")

        fig = px.pie(df, values=df["tipo de evento"].value_counts().tolist(), names=df["tipo de evento"].unique().tolist(),
                     title='Tipo de eventos')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # st.subheader("Regiones")
        fig = px.pie(df, values=df["region"].value_counts().tolist(), names=df["region"].unique().tolist(),
                     title='Regiones involucradas')
        st.plotly_chart(fig, use_container_width=True)

    # Actores involucrados
    df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')

    fig = px.bar(
        df_actor1,
        x="actor principal",
        y="cuenta",
        title="Actores participantes"
    )
    st.plotly_chart(fig, use_container_width=True)

    # evolucion de eventos
    #
    # dt = df.groupby(df.fecha.dt.year)['tipo'].count()
    # dt = df.groupby([(df.fecha.dt.year), (df.fecha.dt.month)])['tipo'].count()

    serial = serie.groupby(
        ['tipo de evento']).size().reset_index(name='cuenta')
    # serial.groupby('tipo')['cuenta'].plot(legend=True, xlabel="")
    fig = px.bar(
        serial,
        x="tipo de evento",
        y="cuenta",
        title="Conflictos en el tiempo"
    )
    fig.update_layout(showlegend=False, xaxis_title='Tipo de evento')
    st.plotly_chart(fig, use_container_width=True)

    # eventos en el tiempo

    # dt = df.groupby([(df.fecha.dt.year), (df.fecha.dt.month)])['tipo'].count()
    # dt.plot(kind='line', figsize=(15, 5), title='Cantidad de eventos por año')

    serial = df.groupby(['fecha', 'tipo de evento']
                        ).size().reset_index(name='cuenta')
    serial['year'] = pd.DatetimeIndex(serial['fecha']).year
    serial['month'] = serial['fecha'].dt.strftime('%m')
    serial['mes'] = serial['fecha'].dt.strftime('%Y-%m')

    serial = serial.groupby(['mes', 'tipo de evento'],
                            as_index=False)['cuenta'].sum()

    figserial = px.line(serial, x="mes", y="cuenta", line_group="tipo de evento",
                        color="tipo de evento", markers=True)
    figserial.update_layout(title='Desarrrollo de eventos en el tiempo',
                            xaxis_title='Fecha',
                            yaxis_title='No. de eventos',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                title="",
                                yanchor="bottom",
                                y=1,
                                xanchor="right",
                                x=1,
                                traceorder="reversed",
                                title_font_family="Times New Roman",
                                font=dict(
                                    family="Courier",
                                    size=11,
                                    color="black"
                                ),
                                # bgcolor="LightSteelBlue",
                                # bordercolor="Black",
                                borderwidth=1
                            ))

    st.plotly_chart(figserial, use_container_width=True)

with tabTable:  # tabla de datos

    st.dataframe(df, height=600)

with tabIA:  # EDA IA
    # col, cor = st.columns(2, gap="medium")
    # with col:
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

    # with cor:
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


with tabBleau:  # graficos personalizados
    report = pgw.walk(df, return_html=True)
    components.html(report, height=1000, scrolling=True)


with tabProfile:  # perfil de datos
    st.write("Perfil de datos")
    # pr = ProfileReport(df, explorative=True, minimal=True)
    # st_profile_report(pr)

    # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    # st.bar_chart(chart_data)
    # import pandas_profiling
    # from streamlit_pandas_profiling import st_profile_report
    # pr = df.profile_report(pr)
    # st: profile_report(pr)

with tabInfo:
    col, cor = st.columns(2, gap="medium")
    with col:
        st.write("El Proyecto de Datos Mapuche (MDP) tiene como objetivos identificar, digitalizar, compilar, procesar y armonizar información cuantitativa respecto al pueblo Mapuche. ")

        st.write("Basados en MACEDA (Mapuche Chilean State Conflict Event Database), primer registro sistemático de eventos relacionados al conflicto entre el pueblo mapuche y el estado chileno.\n")

        st.write("MPD reporta información del conflicto entre el Estado Chileno y el pueblo mapuche. La Base de Datos de Eventos sobre el Conflicto Mapuche-Estado Chileno MACEDA (por su acrónimo en inglés) reporta más de 4500 eventos para el período 1990-2021.")
        st.write("")
        st.markdown(
            "[Descargar datos](https://sites.google.com/view/danyjaimovich/links/mdp)")
        st.write("")
        st.write("Cómo citar:")
        st.write("Cayul, P., A. Corvalan, D. Jaimovich, and M. Pazzona (2022). Introducing MACEDA: New Micro-Data on an Indigenous Self-Determination Conflict. Journal of Peace Research ")

    with cor:
        st.image("images/maceda.jpg", width=500)
