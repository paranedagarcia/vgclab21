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
from pandasai import SmartDatalake  # para multiples dataframes
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
    page_title="Conflictos",
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


def df_filterfecha(message, df):
    dates_selection = st.sidebar.slider('%s' % (message),
                                        min_value=min(df['fecha']),
                                        max_value=max(df['fecha']),
                                        value=(min(df['fecha']), max(df['fecha'])))
    mask = df['fecha'].between(*dates_selection)
    number_of_result = df[mask].shape[0]
    filtered_df = df[mask]
    return filtered_df


local_css("style/style.css")

# carga de datos
# comunas
try:
    dfcomunas = load_data_csv("https://data.vgclab.cl/public_data/comunas.csv")
except:
    st.error("Error al cargar los datos de comunas")
    st.stop()

try:
    datos = "https://data.vgclab.cl/public_data/dataset_conflictos_2008-2020.csv"
    datos = "data/dataset_conflictos_2008-2020.csv"
    df = pd.read_csv(datos, sep=";", encoding="utf-8", na_values=".")
except:
    st.error("Error al cargar los datos")
    st.stop()


df['fecha'] = pd.to_datetime(df[['año', 'mes', 'dia']].astype(str).agg(
    '-'.join, axis=1), errors='coerce', format='mixed', dayfirst=True)
# cambiar fecha al inicio
fecha = df.pop("fecha")
df.insert(1, "fecha", fecha)


# metricas
# years = df["fecha"].dt.year.unique().tolist()
# years.insert(0, "Todos")


# medios
medios = df["medio"].unique().tolist()
medios.insert(0, "Todos")

# sidebar
#
# anual = st.sidebar.selectbox("Seleccione un año", years)

# if anual is not 'Todos':
#    df = df[df["fecha"].dt.year == anual]
# else:
#    pass

medio = st.sidebar.selectbox(
    "Medios", medios)
if medio is not 'Todos':
    df = df[df["medio"] == medio]
else:
    df = df

# termino = st.sidebar.slider("Año de termino", 2008, 2020, 2020)

# TIPO DE CONFLICTO SOCIAL
tipos = df["Tipo de conflicto social"].unique().tolist()
# tipos.insert(0, "Todos")

tipoconflicto = st.sidebar.multiselect(
    "Tipo de conflicto", tipos, default=tipos)
if tipoconflicto is not None:
    df = df[df["Tipo de conflicto social"].isin(tipoconflicto)]
else:
    df = df

heridos_manifestantes = df["Manifestantes heridos"].sum()
heridos_carabineros = df["Carabineros heridos"].sum()
muertos_manifestantes = df["Manifestantes muertos"].sum()
muertos_carabineros = df["Carabineros muertos"].sum()
arrestos = df["Arrestos"].sum()
heridos_personas = df["Personas heridas"].sum()
muertos_personas = df["Personas muertas"].sum()

# main
st.subheader("Evolución de conflictos en Chile (2008 - 2020)")

tabPanel, tabTable, tabIA, tabBleau, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Análisis", "Información"])


with tabPanel:
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Menciones", millify(len(df), precision=2))
    col2.metric("Manifestantes heridos", millify(
        heridos_manifestantes, precision=2))
    col3.metric("Carabineros heridos", millify(
        heridos_carabineros, precision=2))
    col4.metric("Arrestos", millify(arrestos, precision=2))

    col7, col8, col9, col10 = st.columns(4, gap="medium")
    col7.metric("Manifestantes muertos", muertos_manifestantes)
    col8.metric("Carabineros muertos", muertos_carabineros)
    col9.metric("Personas heridas", heridos_personas)
    col10.metric("Personas muertas", muertos_personas)

    style_metric_cards()

    # st.write("Menciones en Medios de Comunicación")
    # mencioines ne medios de comunicacion
    df_medios = df.groupby('medio').size().reset_index(name='cuenta')
    fig = px.bar(
        df_medios,
        x="medio",
        y="cuenta",
        title="Menciones en medios de comunicación"
    )
    fig.update_layout(showlegend=False,
                      xaxis_title=None,
                      yaxis_title=None,
                      # yaxis_tickformat="20,.2f"
                      yaxis=dict(tickformat=",.2r")

                      )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig = px.pie(
            names=["Manifestantes heridos",
                   "Carabineros heridos", "Personas heridas"],
            values=[heridos_manifestantes,
                    heridos_carabineros, heridos_personas],
            title="Manifestantes heridos vs Carabineros heridos vs Personas heridas"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            names=["Manifestantes muertos", "Carabineros muertos"],
            values=[muertos_manifestantes, muertos_carabineros],
            title="Manifestantes muertos vs Carabineros muertos"
        )
        st.plotly_chart(fig, use_container_width=True)

    # col3, col4 = st.columns(2, gap="medium")
    # with col3:
        # ACTOR DEMANDANTE
    df_afectado = df.groupby(
        'Actor demandante').size().reset_index(name='cuenta')

    fig = px.bar(
        df_afectado,
        y="Actor demandante",
        x="cuenta",
        color="Actor demandante",
        title="Actores demandantes",
        orientation="h", height=500
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # with col4:
    # ACTOR DESTINO DE PROTESTA
    df_afectado = df.groupby(
        'Actor destino de protesta').size().reset_index(name='cuenta')

    fig = px.bar(
        df_afectado,
        y="Actor destino de protesta",
        x="cuenta",
        color="Actor destino de protesta",
        title="Actores afectados",
        orientation="h", height=600
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # DESARROLLO DE CONFLICTOS EN EL TIEMPO
    #
    stiempo = df.groupby(['fecha', 'Tipo de conflicto social']
                         ).size().reset_index(name='cuenta')
    stiempo['year'] = pd.DatetimeIndex(stiempo['fecha']).year
    stiempo['month'] = stiempo['fecha'].dt.strftime('%m')
    stiempo['mes'] = stiempo['fecha'].dt.strftime('%Y-%m')

    stiempo = stiempo.groupby(['mes', 'Tipo de conflicto social'],
                              as_index=False)['cuenta'].sum()

    figserial = px.line(stiempo, x="mes", y="cuenta", line_group="Tipo de conflicto social",
                        color="Tipo de conflicto social", markers=True)
    figserial.update_layout(title='Desarrrollo de conflictos en el tiempo',
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

with tabTable:

    # procesos

    st.dataframe(df, height=500)

with tabIA:  # EDA IA
    st.subheader("Análisis exploratorio con Inteligencia Artificial")
    with st.expander("Información importante"):
        st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el conjunto de datos. Ingrese su consulta la que pudiera ser respondida por el modelo en forma de texto o una imagen gráfica.")
        st.write(
            "Por ejemplo, puede preguntar: ¿Cuántos eventos de tipo 'X' ocurrieron en la región 'Y' en el año '2018'?")
        st.warning(
            "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")
    st.write("")

    user_path = os.getcwd()
    # llm = OpenAI(api_token=API_KEY)
    llm = OpenAI(client=OpenAI, streaming=True,
                 api_token=API_KEY, temperature=0.5)

    # agent = create_csv_agent(OpenAI(temperature=0.5), datos)

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

with tabInfo:
    st.write("Información")


# eliminar columnas
# df = df.drop(columns=["p1", "p2a", "p2b", "p2c", "pm",
#                       "p3", "p5d", "p5e", "p5f",  "p11a",
#                       "p13b", "p13c", "p13d", "p17b", "p17c", "p17d", "p17e", "p17f", "p18b",
#                       'p19a1', 'p19a2', 'p19b1', 'p19b2', 'p19c1', 'p19c2', 'p19d1', 'p19d2', 'p19e1', 'p19e2',
#                       "p24", "p26b",
#                       "p26d", "p26f", "p28b", "p28d", "p28f", "p29a", "p29b", "p29c", "p29d", "p29e", "p29f"])

# cambio de nombre de columnas
# df = df.rename(columns={
#     'Region': 'Region_id',
#     'Provincia': 'Provincia_id',
#     'Comuna': 'Comuna_id',
#     'pa': 'Tipo medio',
#     'pb': 'Radial',
#     'p0': 'Cobertura',
#     'p3a': 'Tipo noticia',
#     'p4': 'lineas',
#     'p5a': 'dia',
#     'p5b': 'mes',
#     'p5c': 'año',
#     'p6': 'Region',
#     'p7': 'Provincia',
#     'p8': 'Comuna',
#     'p9': 'Localidad',
#     'p9a': 'Urbano',
#     'p10': 'Lugar objetivo',
#     'p10a': 'ID evento',
#     'p11': 'Cantidad participantes',
#     'p12': 'Estimacion',
#     'p13a': 'Grupo social',
#     'p16': 'Organizaciones participantes',
#     'p17a': 'Actor',
#     'p20a': 'Conflicto social',
#     'p20b': 'Conflicto institucional',
#     'p21': 'Carabineros',
#     'p23': 'Arrestos',
#     'p25': 'Heridos',
#     'p26a': 'Manifestantes heridos',
#     'p26c': 'Carabineros heridos',
#     'p26e': 'Personas heridas',
#     'p28a': 'Manifestantes muertos',
#     'p28c': 'Carabineros muertos',
#     'p28e': 'Personas muertas'
# })
