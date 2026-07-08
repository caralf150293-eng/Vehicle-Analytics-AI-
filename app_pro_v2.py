
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Vehicle Analytics AI",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# RUTAS
# ==========================================

#ruta_proyecto = "/content/drive/MyDrive/Proyecto_Final_Automoviles"

#ruta_csv = f"{ruta_proyecto}/data/Car_sales-selected-columns.csv"
#ruta_logo = f"{ruta_proyecto}/images/logo.png"
#ruta_banner = f"{ruta_proyecto}/images/banner.png"

ruta_csv = "datos/Car_sales-selected-columns.csv"

ruta_logo = "imagenes/logo.png"

ruta_banner = "imagenes/banner.png"

# ==========================================
# CARGA DE DATOS
# ==========================================

df = pd.read_csv(ruta_csv)

columnas_numericas = [
    "Sales in thousands",
    "4-year resale value",
    "Price in thousands",
    "Engine size",
    "Horsepower",
    "Wheelbase",
    "Width"
]

for columna in columnas_numericas:
    df[columna] = pd.to_numeric(df[columna], errors="coerce")

df = df.dropna()

# ==========================================
# MODELO ML
# ==========================================

X = df[
    [
        "Price in thousands",
        "Engine size",
        "Horsepower",
        "Wheelbase",
        "Width"
    ]
]

y = df["Vehicle type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ==========================================
# VARIABLES EJECUTIVAS
# ==========================================

tipo_principal = df["Vehicle type"].value_counts().idxmax()
fabricante_top = df.groupby("Manufacturer")["Sales in thousands"].sum().idxmax()
precio_promedio = round(df["Price in thousands"].mean(), 2)
hp_promedio = round(df["Horsepower"].mean(), 2)
motor_promedio = round(df["Engine size"].mean(), 2)

hora_actual = datetime.now().hour

if hora_actual < 12:
    saludo = "Buenos días"
    emoji_saludo = "☀️"
elif hora_actual < 19:
    saludo = "Buenas tardes"
    emoji_saludo = "🌤️"
else:
    saludo = "Buenas noches"
    emoji_saludo = "🌙"

fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071A2C, #0A3D62);
}

[data-testid="stSidebar"] * {
    color: white;
}

.top-bar {
    background: linear-gradient(90deg, #071A2C, #0A3D62);
    padding: 18px 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

.top-title {
    font-size: 25px;
    font-weight: 900;
}

.top-subtitle {
    font-size: 14px;
    opacity: 0.85;
}

.status-pill {
    background-color: #16a085;
    color: white;
    padding: 8px 14px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 700;
    text-align: center;
}

.main-title {
    font-size: 46px;
    font-weight: 900;
    color: #0A3D62;
    text-align: center;
    margin-top: 8px;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 21px;
    color: #3C6382;
    text-align: center;
    margin-bottom: 30px;
}

.kpi-card {
    background: linear-gradient(135deg, #0A3D62, #1E6091);
    padding: 24px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    min-height: 165px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.25);
}

.kpi-icon {
    font-size: 35px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 34px;
    font-weight: 900;
}

.kpi-label {
    font-size: 15px;
    color: #EAF4FF;
}

.kpi-note {
    margin-top: 8px;
    font-size: 13px;
    color: #BFE9FF;
}

.card {
    background-color: white;
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.section-title {
    color: #0A3D62;
    font-size: 24px;
    font-weight: 900;
    margin-bottom: 12px;
}

.insight {
    background-color: #F4F9FC;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    border-left: 5px solid #0A3D62;
    font-size: 16px;
}

.model-card {
    background: linear-gradient(135deg, #ECF0F1, #FFFFFF);
    border: 1px solid #DDE6ED;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.result-number {
    font-size: 36px;
    font-weight: 900;
    color: #0A3D62;
}

.footer {
    text-align: center;
    margin-top: 35px;
    color: #7f8c8d;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

logo = Image.open(ruta_logo)

st.sidebar.image(logo, use_container_width=True)

st.sidebar.markdown("## Vehicle Analytics AI")
st.sidebar.markdown("**Proyecto Final**")
st.sidebar.markdown("Diplomado IA 2026")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Menú principal",
    [
        "🏠 Inicio",
        "📊 Dashboard Ejecutivo",
        "📈 Análisis Exploratorio",
        "🔥 Correlación",
        "🤖 Modelo Predictivo",
        "🎯 Predicción",
        "📄 Reporte Ejecutivo"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(f"""
**Dataset:** Car Sales  
**Registros:** {df.shape[0]}  
**Modelo:** Árbol de Decisión  
**Estado:** Activo  
**Versión:** 2.0
""")

# ==========================================
# INICIO EJECUTIVO
# ==========================================

if menu == "🏠 Inicio":

    col_top1, col_top2 = st.columns([3, 1])

    with col_top1:
        st.markdown(f"""
        <div class="top-bar">
            <div class="top-title">🚗 Vehicle Analytics AI</div>
            <div class="top-subtitle">Proyecto Final | Carlos Alfredo González Hernández | Diplomado IA 2026</div>
        </div>
        """, unsafe_allow_html=True)

    with col_top2:
        st.markdown(f"""
        <div class="top-bar">
            <div class="status-pill">🟢 Sistema Activo</div>
            <div class="top-subtitle" style="text-align:center; margin-top:10px;">{fecha_actual}</div>
        </div>
        """, unsafe_allow_html=True)

    banner = Image.open(ruta_banner)
    st.image(banner, use_container_width=True)

    st.markdown(f'<div class="main-title">{saludo}, Carlos {emoji_saludo}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Centro ejecutivo para análisis, clasificación y predicción de vehículos</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🚗</div>
            <div class="kpi-value">{df.shape[0]}</div>
            <div class="kpi-label">Vehículos registrados</div>
            <div class="kpi-note">✔ Dataset cargado correctamente</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🏭</div>
            <div class="kpi-value">{df["Manufacturer"].nunique()}</div>
            <div class="kpi-label">Fabricantes</div>
            <div class="kpi-note">✔ Marcas disponibles</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💰</div>
            <div class="kpi-value">{precio_promedio}</div>
            <div class="kpi-label">Precio promedio</div>
            <div class="kpi-note">Miles de dólares</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⚙️</div>
            <div class="kpi-value">{hp_promedio}</div>
            <div class="kpi-label">Horsepower promedio</div>
            <div class="kpi-note">Potencia promedio</div>
        </div>
        """, unsafe_allow_html=True)

    graf1, graf2 = st.columns(2)

    with graf1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Top 10 fabricantes por ventas</div>', unsafe_allow_html=True)

        ventas = (
            df.groupby("Manufacturer")["Sales in thousands"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            ventas,
            x="Sales in thousands",
            y="Manufacturer",
            orientation="h",
            title="Ventas en miles por fabricante",
            labels={
                "Sales in thousands": "Ventas en miles",
                "Manufacturer": "Fabricante"
            }
        )

        fig.update_layout(
            height=420,
            yaxis=dict(autorange="reversed"),
            title_x=0.02
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with graf2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚘 Distribución por tipo de vehículo</div>', unsafe_allow_html=True)

        tipos = df["Vehicle type"].value_counts().reset_index()
        tipos.columns = ["Vehicle type", "Cantidad"]

        fig = px.pie(
            tipos,
            names="Vehicle type",
            values="Cantidad",
            title="Participación por categoría"
        )

        fig.update_layout(height=420, title_x=0.02)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📌 Resumen ejecutivo</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="insight">✔ El dataset contiene <b>{df.shape[0]}</b> vehículos.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ Se identificaron <b>{df["Manufacturer"].nunique()}</b> fabricantes.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ El tipo predominante es <b>{tipo_principal}</b>.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ El fabricante con mayores ventas es <b>{fabricante_top}</b>.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ El precio promedio es de <b>{precio_promedio}</b> miles de dólares.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 Estado del modelo</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f'<div class="result-number">{accuracy:.2f}</div>', unsafe_allow_html=True)
            st.write("Accuracy")

        with m2:
            st.markdown(f'<div class="result-number">{precision:.2f}</div>', unsafe_allow_html=True)
            st.write("Precision")

        with m3:
            st.markdown(f'<div class="result-number">{recall:.2f}</div>', unsafe_allow_html=True)
            st.write("Recall")

        with m4:
            st.markdown(f'<div class="result-number">{f1:.2f}</div>', unsafe_allow_html=True)
            st.write("F1 Score")

        st.progress(float(accuracy))

        st.write("""
        **Modelo utilizado:** Árbol de Decisión  
        **Variable objetivo:** Vehicle type  
        **Variables predictoras:** precio, motor, horsepower, wheelbase y width.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Vehicle Analytics AI | Versión 2.0 | Proyecto Final | Carlos Alfredo González Hernández | Desarrollado con Python + Streamlit
    </div>
    """, unsafe_allow_html=True)


elif menu == "📊 Dashboard Ejecutivo":

    st.title("📊 Dashboard Ejecutivo")

    c1, c2, c3 = st.columns(3)

    c1.metric("Precio promedio", precio_promedio)
    c2.metric("Motor promedio", motor_promedio)
    c3.metric("HP promedio", hp_promedio)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        ventas = df.groupby("Manufacturer")["Sales in thousands"].sum().sort_values(ascending=False).head(10).reset_index()

        fig = px.bar(
            ventas,
            x="Manufacturer",
            y="Sales in thousands",
            title="Top 10 fabricantes por ventas",
            labels={"Sales in thousands": "Ventas en miles"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="Price in thousands",
            title="Distribución de precios",
            labels={"Price in thousands": "Precio en miles"}
        )

        st.plotly_chart(fig, use_container_width=True)


elif menu == "📈 Análisis Exploratorio":

    st.title("📈 Análisis Exploratorio")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="Horsepower",
            y="Price in thousands",
            color="Vehicle type",
            title="Precio vs Horsepower"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="Engine size",
            y="Price in thousands",
            color="Vehicle type",
            title="Precio vs Tamaño del motor"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            df["Vehicle type"].value_counts().reset_index(),
            x="Vehicle type",
            y="count",
            title="Cantidad por tipo de vehículo"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df,
            x="Vehicle type",
            y="Horsepower",
            title="Distribución de horsepower por tipo"
        )

        st.plotly_chart(fig, use_container_width=True)


elif menu == "🔥 Correlación":

    st.title("🔥 Matriz de Correlación")

    correlacion = df[columnas_numericas].corr()

    fig = px.imshow(
        correlacion,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Matriz de correlación de variables numéricas"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    La matriz de correlación permite identificar las relaciones entre las variables numéricas.
    Se observa que el precio se relaciona principalmente con el valor de reventa, horsepower y tamaño del motor.
    """)


elif menu == "🤖 Modelo Predictivo":

    st.title("🤖 Modelo Predictivo")

    st.write("""
    Se entrenó un modelo de **Árbol de Decisión** para clasificar el tipo de vehículo
    utilizando variables numéricas como precio, motor, horsepower, wheelbase y width.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.markdown("---")

    matriz = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        matriz,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicción", y="Valor real", color="Cantidad"),
        x=modelo.classes_,
        y=modelo.classes_,
        title="Matriz de Confusión"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    El modelo presenta un desempeño adecuado. Sin embargo, puede cometer algunos errores debido
    al desbalance entre las clases Passenger y Car.
    """)


elif menu == "🎯 Predicción":

    st.title("🎯 Predicción de Tipo de Vehículo")

    st.write("Ingresa las características del vehículo para obtener una predicción del modelo.")

    col1, col2 = st.columns(2)

    with col1:
        precio = st.number_input("Precio en miles de dólares", min_value=0.0, value=35.0)
        motor = st.number_input("Tamaño del motor", min_value=0.0, value=2.5)
        horsepower = st.number_input("Horsepower", min_value=0, value=210)

    with col2:
        wheelbase = st.number_input("Wheelbase", min_value=0.0, value=110.0)
        width = st.number_input("Width", min_value=0.0, value=72.0)

    nuevo = pd.DataFrame({
        "Price in thousands": [precio],
        "Engine size": [motor],
        "Horsepower": [horsepower],
        "Wheelbase": [wheelbase],
        "Width": [width]
    })

    st.subheader("Datos ingresados")
    st.dataframe(nuevo)

    if st.button("Predecir tipo de vehículo"):

        prediccion = modelo.predict(nuevo)[0]

        st.markdown(f"""
        <div class="model-card">
            <h2>Resultado de la predicción</h2>
            <div class="result-number">{prediccion}</div>
            <p>El modelo clasifica este vehículo dentro de la categoría mostrada.</p>
        </div>
        """, unsafe_allow_html=True)


elif menu == "📄 Reporte Ejecutivo":

    st.title("📄 Reporte Ejecutivo")

    st.markdown(f"""
    ## Conclusiones generales

    El proyecto permitió desarrollar una solución completa de análisis de datos y Machine Learning
    aplicada a un conjunto de datos de automóviles.

    **Hallazgos principales:**

    - El dataset contiene **{df.shape[0]} vehículos** y **{df["Manufacturer"].nunique()} fabricantes**.
    - El tipo de vehículo predominante es **{tipo_principal}**.
    - El fabricante con mayores ventas es **{fabricante_top}**.
    - El precio promedio de los vehículos es de **{precio_promedio} miles de dólares**.
    - Se identificaron relaciones importantes entre precio, valor de reventa, horsepower y tamaño del motor.
    - El modelo de Árbol de Decisión permitió clasificar el tipo de vehículo con métricas aceptables.

    ## Cierre

    Esta aplicación integra análisis exploratorio, visualización interactiva, evaluación del modelo
    y predicción de nuevos datos, cumpliendo con los elementos principales del proyecto final.
    """)

    st.markdown("""
    <div class="footer">
        Vehicle Analytics AI | Proyecto Final | Carlos Alfredo González Hernández | Diplomado IA 2026
    </div>
    """, unsafe_allow_html=True)
