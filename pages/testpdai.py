#
import os
import streamlit as st
import pandas as pd
# from pandasai import PandasAI
from pandasai import SmartDataframe
from pandasai import SmartDatalake  # para multiples dataframes
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import matplotlib.pyplot as plt
from datetime import datetime
from funciones import load_data_csv
from dotenv import load_dotenv

load_dotenv()

# configuration
st.set_page_config(
    page_title="Maceda",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

user_path = os.getcwd()
# Instantiate a LLM
llm = OpenAI(api_token="OPENAI_API_TOKEN")


df = load_data_csv("https://data.vgclab.cl/public_data/dataset_maceda.csv")

# df = pd.read_csv(datos, sep=",", low_memory=False)
# sacar columnas
# df = df.drop(columns=["id", "id_evento", "fecha_reportada"])

df = df.drop(columns=["id_evento"])
df = df.rename(columns={
    'fecha_reportada': "fecha informada",
    'ubicacion_tipo': 'ubicacion',
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

with st.expander("Mostrar las primeras 15 observaciones"):
    st.write(df.head(15))

prompt = st.text_area("Genera la consulta")
# st.write(query)
if st.button("Generar respuesta"):
    if prompt:
        llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
        # query = SmartDataframe(df, config={"llm": llm})
        query = Agent(df, config={"llm": llm,
                                  "save_charts": True,
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
