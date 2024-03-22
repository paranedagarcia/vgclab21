#
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import openai
import numpy as np
import datetime as dt
from st_pages import Page, show_pages, add_page_title
from PIL import Image

# configuration
st.set_page_config(
    page_title="VGCLAB",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# local css


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")
image = Image.open('images/vgclab-negro.jpg')

# --------------------------
# sidebar
# --------------------------
# st.sidebar.image(image)


st.sidebar.write("[Fundación Chile 21](https://chile21.cl/)")
st.sidebar.divider()
st.sidebar.write("San Sebastián 2807, Las Condes, Santiago de Chile")

show_pages([Page("app.py", "Inicio"),
            Page("pages/maceda.py", "Conflictos mapuches"),
            Page("pages/conflictos.py", "Conflictos territoriales"),
            Page("pages/justicia.py", "Eficiencia de la Justicia"),
            Page("pages/salud.py", "Salud mental"),
            Page("pages/testpdai.py", "testpdai"),
            Page("pages/clientes.py", "Segmentación de clientes"),
            Page("pages/ventas.py", "Ventas")
            # Page("pages/pinguinos.py", "Pinguinos"),
            # Page("pages/descuentos.py", "Descuentos"),
            # Page("pages/trees.py", "Arboles"),
            # Page("pages/planilla.py", "Google Sheet"),
            # Page("pages/excel.py", "Excel")
            ])

st.header("Fundación Chile 21")
st.subheader("Plataforma de datos")
st.divider()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.image(image, width=200)
with col2:
    st.write("El Laboratorio sobre Violencias y Gestión de Conflictos VGC LAB es un espacio de reflexión, investigación colaborativa y generación de propuestas de intervención social para abordar la violencia y los conflictos en diferentes ámbitos, con el fin de promover una cultura de la paz y buena gestión de los conflictos.")

with col3:
    image = Image.open('images/maceda.jpg')
    st.image(image, width=400)
