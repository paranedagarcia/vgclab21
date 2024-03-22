# goggle sheet public
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# configuration
st.set_page_config(
    page_title="Descuentos",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# url = "https://docs.google.com/spreadsheets/d/1P99NTcjhGtWaLn2i5_LISWSaUnDRrdvOEGasaxHolqk/edit?usp=sharing"
# conn = st.connection("gsheets", type=GSheetsConnection)
# df = conn.read(spreadsheet=url)

url = "https://docs.google.com/spreadsheets/d/1AOdI3PwJ_Z3AfwITlZYX8Fh7BV59AtZEc3mDUlssV3Y/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(spreadsheet=url, header=4, usecols=[
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

proyectos = df["Proyecto"].unique().tolist()
proyectos.insert(0, "Todos")

# unidades = df["PRODUCTO"].unique().tolist()
# unidades.insert(0, "Todos")
valor_pie = [.10, .15, .20, .25, .30, .35, .40, .45, .50]
financiamientos = [.8, .85, .90]
creditos = [20, 25, 30]
bonopies = [0, .05, .10, .15, .20, .25, .30]

selected_proyecto = st.sidebar.selectbox(
    "Seleccione Proyecto",
    proyectos
)

# funciones auxiliares
# funcion para mostar un valor decimal en porcentaje sin decimales


def integer_to_percentage(decimal_value):
    percentage = decimal_value * 100
    return round(percentage)


def decimal_to_percentage(decimal_value):
    percentage = decimal_value * 100
    return (percentage)


def tasa_mes():
    tasa_mensual = (1+tasa_anual)**(1/12)-1
    st.session_state['tasames'] = (1 + st.session_state['tasaanual'])**(1/12)-1
    # return tasa_mensual


def simulacion():
    return 200


# filter dataframe
if selected_proyecto != "Todos":
    df = df[df["Proyecto"] == selected_proyecto]
    unidades = df["Depto"].unique().tolist()
    unidades.insert(0, "Todos")
    selected_unidad = st.sidebar.selectbox(
        "Seleccione Unidad",
        unidades
    )
    if selected_unidad != "Todos":
        df = df[df["Depto"] == selected_unidad]

# datos de producto
tipologia = df["Tipologia"].iloc[0]
sup_venta = df["Sup vta"].iloc[0]
sup_total = df["Metraje"].iloc[0]
orientacion = df["Orientacion"].iloc[0]
precio_lista = (df["Precio Lista"].iloc[0]) * 1000
precio = df["Precio Dscto3"].iloc[0] * 1000

st.sidebar.write(
    f"Tipologia: <span class='right'>{tipologia}</span>", unsafe_allow_html=True)
st.sidebar.write(
    f"Sup. vta.: <span class='right'>{sup_venta}</span>", unsafe_allow_html=True)
st.sidebar.write(
    f"Metraje: <span class='right'>{sup_total}</span>", unsafe_allow_html=True)
st.sidebar.write(
    f"Orientación: <span class='right'>{orientacion}</span>", unsafe_allow_html=True)
st.sidebar.write(
    f"Precio de lista: <span class='right'>{precio_lista}</span>", unsafe_allow_html=True)
st.sidebar.write(
    f"Precio descuento: <span class='right'>{precio}</span>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Simulador de descuentos", "Tabla de descuentos"])


with tab1:
    # parametros
    st.subheader("Simulador de descuentos")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pie = st.selectbox("Pie (%)", valor_pie,
                               format_func=integer_to_percentage)
        with col2:
            bonopie = st.selectbox("Bono pie (%)", bonopies,
                                   format_func=integer_to_percentage)
        with col3:
            financiamiento = st.selectbox(
                "Financiamiento (%)", financiamientos, format_func=integer_to_percentage)
        with col4:
            credito_years = st.selectbox("Años de crédito", creditos)

    # st.write(st.session_state)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            meses = st.number_input(
                "Plazo en meses", value=24, min_value=0, max_value=60, step=1)
        with col2:
            tasa_anual = st.number_input(
                "Tasa anual", key="tasaanual", value=5.50, step=.01, format="%.2f",
                on_change=tasa_mes)
        with col3:
            tasa_mensual = st.number_input(
                "Tasa mensual", key="tasames",  step=.5, format="%.4f")
        with col4:
            st.button("Calcular", key="calcular", type="primary",
                      on_click=simulacion, use_container_width=True)

    # st.info(f"selectbox returned: {tasa_mensual}")
    with st.container():
        valor_pie = 1000
        valor_credito = 3450
        valor_dividendo = 230
        retorno = 425.7

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(
                f"Valor pie: <span class='left'>{valor_pie}</span>", unsafe_allow_html=True)
        with col2:
            st.write(
                f"Valor credito: <span class='left'>{valor_credito}</span>", unsafe_allow_html=True)
        with col3:
            st.write(
                f"Valor dividendo: <span class='left'>{valor_dividendo}</span>", unsafe_allow_html=True)
        with col4:
            st.write(
                f"Retorno: <span class='left'>{retorno}</span>", unsafe_allow_html=True)

    # st.divider()
    # simulador de alternativas
    with st.container():
        # resultados
        arriendo = 100
        kit_inversionista = 100
        gastos_inversionista = 100
        dividendo = 100
        kit_propietario = 199
        gastos_propietario = 100
        gc = 100
        #
        st.subheader("Simulación de alternativas")
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        with col1:
            st.markdown("**Arriendo garantizado 2 años**")
            st.write(
                f"Arriendo: <span class='right'>{arriendo}</span>", unsafe_allow_html=True)
            st.write(
                f"Valor kit inversionista: <span class='right'>{kit_inversionista}</span>", unsafe_allow_html=True)
            st.write(
                f"Gastos operacionales: <span class='right'>{gastos_inversionista}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Dividendo garantizado 2 años**")
            st.write(
                f"Dividendo: <span class='right'>{dividendo}</span>", unsafe_allow_html=True)
            st.write(
                f"Valor kit propietario: <span class='right'>{kit_propietario}</span>", unsafe_allow_html=True)
            st.write(
                f"Gastos operacionales: <span class='right'>{gastos_propietario}</span>", unsafe_allow_html=True)
            st.write(
                f"GC (3 meses): <span class='right'>{gc}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("**Descuento máximo de la unidad**")
        with col4:
            st.markdown("**Financiamiento Imagina-pie**")

    # nuevo precio de venta
    with st.container():
        nuevo_precio = 0
        st.subheader("Nuevo precio de venta")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(
                f"Con promoción: <span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f"Dcto comercial realizado UF:<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f"Dcto comercial realizado %: <span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
        with col2:
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
        with col3:
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
        with col4:
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)
            st.write(
                f".<span class='right'>{nuevo_precio}</span>", unsafe_allow_html=True)

with tab2:
    st.dataframe(df)
