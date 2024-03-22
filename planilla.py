# goggle sheet descuentos
import streamlit as st
from streamlit_gsheets import GSheetsConnection

url = "https://docs.google.com/spreadsheets/d/1AOdI3PwJ_Z3AfwITlZYX8Fh7BV59AtZEc3mDUlssV3Y/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(spreadsheet=url, header=4, usecols=[
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

st.dataframe(df)
