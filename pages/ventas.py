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
import sqlalchemy as sal
from sqlalchemy import create_engine, text, select

# configuration
st.set_page_config(
    page_title="Eficiencia de justicia",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("Generación e interpretación de ventas")

df = load_data_csv("data/operaciones.csv")

# connect to SQL Server
# conectar a SQL
server = "imaginaazure.database.windows.net"
database = "imagina-dw-reportes"
username = "admin_azure"
password = "imagina_DW"
conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server + \
    ';DATABASE='+database+';AUTOCOMMIT=FALSE;UID='+username+';PWD=' + password
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={conn_str}", fast_executemany=True)

# carga de mkt
df_mkt = pd.read_sql("SELECT top(1000) * FROM [dbo].[stg_cubit_mkt]", engine)

# carga de operaciones
df_opera = pd.read_sql(
    "SELECT top(1000) * FROM [dbo].[stg_cubit_operaciones]", engine)

with st.expander("Mostrar las primeras observaciones de Marketing"):
    st.write(df_mkt.head(10))

with st.expander("Mostrar las primeras observaciones de Operaciones"):
    st.write(df_opera.head(10))

llm = OpenAI(api_token="OPENAI_API_TOKEN")
prompt = st.text_area("Genera la consulta")
if st.button("Generar respuesta"):
    if prompt:
        llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
        query = SmartDatalake([df_mkt, df_opera], config={"llm": llm,
                                                          "save_charts": True,
                                                          # "save_charts_path": user_path,
                                                          "open-charts": True,
                                                          "verbose": True,
                                                          "response_parser": StreamlitResponse
                                                          })

        # pandas_ai = PandasAI(llm)

        response = query.chat(prompt)
        # response = pandas_ai.run(df, prompt=prompt, streamlit=True)
        st.write(response)
        # st.image(response)

    else:
        st.write("Por favor ingrese una consulta.")
