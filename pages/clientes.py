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
# database
import pyodbc as odbc
import sqlalchemy as sa
from sqlalchemy import create_engine, text, select

# configuration
st.set_page_config(
    page_title="Eficiencia de justicia",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("Segmentation of Clientes")

df = load_data_csv("data/operaciones.csv")

# connect to SQL Server

# cambio de nombres de columnas
df = df.rename(columns={"TipoProducto": "Tipo de producto",
                        "EstadoEtapa": "Estado de etapa",
                        "NombreCliente": "Nombre de cliente",
                        "CorreoElectronico": "Correo",
                        "RutCliente": "Rut",
                        "FechaNacimiento": "Fecha de nacimiento",
                        "EstadoCivil": "Estado civil",
                        "NroHijos": "Número de hijos",
                        "TipoModelo": "Tipo de modelo",
                        "Disponble": "Disponible",
                        "MetrosCuadrados": "Metros cuadrados",
                        "MetrosTerraza": "Metros de terraza",
                        "MetrosInterior": "Metros interior",
                        "MetrosJardin": "Metros de jardín",
                        "EstadoProducto": "Estado de producto",
                        "FechaReserva": "Fecha de reserva",
                        "FechaAprobacionReserva": "Fecha de aprobación de reserva",
                        "FechaFirmaPromesa": "Fecha de firma de promesa",
                        "FechaEscritura": "Fecha de escritura",
                        "FechaAgendada": "Fecha agendada",
                        "FechaEntrega": "Fecha de entrega",
                        "NombreBroker": "Nombre de broker",
                        "PrecioLista": "Precio de lista",
                        "PrecioVenta": "Precio de venta",
                        "PrecioSecundario": "Precio secundario",
                        "FechaCotizacion": "Fecha de cotización",
                        "UsuarioCotizado": "Usuario cotizador",
                        "UsuarioReserva": "Usuario de reserva",
                        "ValorReserva": "Valor de reserva",
                        "ValorPromesa": "Valor de promesa",
                        "ValorPie": "Valor de pie",
                        "ValorAhorroplus": "Valor de ahorro plus",
                        "ValorSubsidios": "Valor de subsidios",
                        "ValorAhorro": "Valor de ahorro",
                        "ValorCredito": "Valor de crédito",
                        })

# with st.expander("Mostrar las primeras observaciones"):

with st.expander("Mostrar las primeras observaciones"):
    st.write(df.head(10))


llm = OpenAI(api_token="OPENAI_API_TOKEN")
prompt = st.text_area("Genera la consulta")
if st.button("Generar respuesta"):
    if prompt:
        llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
        query = SmartDatalake(df, config={"llm": llm,
                                          "save_charts": True,
                                          # "save_charts_path": user_path,
                                          "open-charts": True,
                                          "verbose": True,
                                          "response_parser": StreamlitResponse
                                          })

        # pandas_ai = PandasAI(llm)

        response = query.chat(prompt)
        # response = pandas_ai.run(df, prompt=prompt, streamlit=True)

        if response.endswith("png"):
            st.image(response)
        else:
            st.write(response)

    else:
        st.write("Por favor ingrese una consulta.")
