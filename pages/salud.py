# salud .py
'''

'''
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
user_path = os.getcwd()
# Instantiate a LLM
llm = OpenAI(api_token="OPENAI_API_TOKEN")


df = load_data_csv(
    "https://data.vgclab.cl/public_data/dataset_licencias-2018-2021.csv")

tab1, tab2, tab3, tab4 = st.tabs(["Panel", "Tabla", "IA-EDA", "Análisis"])

with tab1:
    st.write("Panel")
    st.write(df.head(15))
with tab2:
    st.write("Tabla")

with tab3:
    st.write("IA-EDA")
    # Create a SmartDataframe
    sdf = SmartDataframe(df)
    # Create a SmartDatalake
    sdl = SmartDatalake()
    # Add the SmartDataframe to the SmartDatalake
    sdl.add_dataframe(sdf)
    # Create an Agent
    agent = Agent(llm, sdl)
    # Create a StreamlitResponse
    response = StreamlitResponse()
    # Run the Agent
    agent.run(response)
    # Print the response
    response.print()

with tab4:
    st.write("Análisis")


st.write(df.head(15))
