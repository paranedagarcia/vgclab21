

from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
#
# from langchain.llms import OpenAI
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']

# llm = OpenAI(api_token=API_KEY)
llm = OpenAI(client=OpenAI, streaming=True,
             api_token=API_KEY, temperature=0.5)
# pandas_ai = PandasAI(llm)

st.title("Hello World")
file_uploaded = st.file_uploader("Upload a file", type=["csv"])

if file_uploaded is not None:
    data = pd.read_csv(file_uploaded, sep=";", low_memory=False)
    st.write(data.head(5))

    prompt = st.text_area("Ingrese su consulta:")

    if st.button("Generar respuesta"):
        if prompt:
            with st.spinner("Generando respuesta... por favor espere."):
                sdf = SmartDataframe(data, config={"llm": llm})
                st.write(sdf.chat(prompt))
        else:
            st.write("Por favor ingrese una consulta.")

# LLM
