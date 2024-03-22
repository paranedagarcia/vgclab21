#
import streamlit as st
# import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as dt

# tree map

st.title("SF Trees")
st.write(
    "This app analyses trees in San Francisco using"
    " a dataset kindly provided by SF DPW"
)
graph_color = st.sidebar.color_picker("Seleccione n color", "#BC0612")

trees_df = pd.read_csv("data/trees.csv")

df_dbh_grouped = pd.DataFrame(trees_df.groupby(["dbh"]).count()["tree_id"])
df_dbh_grouped.columns = ["tree_count"]

fig = px.histogram(trees_df["dbh"],
                   # title="Edad de los árboles",
                   color_discrete_sequence=[graph_color])

# plotly bar chart
tab1, tab2 = st.tabs(["Charts", "Data table"])
with tab1:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.bar_chart(df_dbh_grouped)
with tab2:
    st.write(trees_df.head())

st.write("Diseño en columnas")
# columnas
col1, col2 = st.columns(2, gap="medium")
# tab3, tab4 = st.tabs(["Bar chart", "Plotly chart"])
with col1:
    st.line_chart(df_dbh_grouped)
with col2:
    # with tab3:
    #    st.bar_chart(df_dbh_grouped)
    # with tab4:
    st.plotly_chart(fig, use_container_width=True)
