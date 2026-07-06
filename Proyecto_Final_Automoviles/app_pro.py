
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Vehicle Analytics AI",
    page_icon="🚗",
    layout="wide"
)

ruta_proyecto = "/content/drive/MyDrive/Proyecto_Final_Automoviles"
ruta_csv = f"{ruta_proyecto}/data/Car_sales-selected-columns.csv"
ruta_logo = f"{ruta_proyecto}/images/logo.png"
ruta_banner = f"{ruta_proyecto}/images/banner.png"

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

X = df[[
    "Price in thousands",
    "Engine size",
    "Horsepower",
    "Wheelbase",
    "Width"
]]

y = df["Vehicle type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

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
    min-height: 155px;
}

.kpi-icon {
    font-size: 34px;
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

.card {
    background-color: white;
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    margin-top: 18px;
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

# Sidebar
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
        "📊 Dashboard",
        "📈 Visualizaciones",
        "🔥 Correlación",
        "🌳 Machine Learning",
        "🎯 Predicción",
        "📄 Conclusiones"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset:** Car Sales  
**Registros:** 157  
**Modelo:** Árbol de Decisión  
**Versión:** 1.0
""")

# HOME EJECUTIVO
if menu == "🏠 Inicio":

    banner = Image.open(ruta_banner)
    st.image(banner, use_container_width=True)

    st.markdown('<div class="main-title">Vehicle Analytics AI</div>', unsafe_allow_html=True)
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
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🏭</div>
            <div class="kpi-value">{df["Manufacturer"].nunique()}</div>
            <div class="kpi-label">Fabricantes</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💰</div>
            <div class="kpi-value">{round(df["Price in thousands"].mean(), 2)}</div>
            <div class="kpi-label">Precio promedio</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⚙️</div>
            <div class="kpi-value">{round(df["Horsepower"].mean(), 2)}</div>
            <div class="kpi-label">Horsepower promedio</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    graf1, graf2 = st.columns(2)

    with graf1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Ventas por fabricante")
        ventas = df.groupby("Manufacturer")["Sales in thousands"].sum().sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ventas.plot(kind="barh", ax=ax)
        ax.set_xlabel("Ventas en miles")
        ax.set_ylabel("Fabricante")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with graf2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚘 Distribución por tipo de vehículo")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Vehicle type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Insights rápidos")

        tipo_principal = df["Vehicle type"].value_counts().idxmax()
        fabricante_top = df.groupby("Manufacturer")["Sales in thousands"].sum().idxmax()

        st.markdown(f'<div class="insight">✔ Tipo predominante: <b>{tipo_principal}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ Fabricante con mayores ventas: <b>{fabricante_top}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ Precio promedio: <b>{round(df["Price in thousands"].mean(), 2)}</b> miles</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight">✔ Potencia promedio: <b>{round(df["Horsepower"].mean(), 2)}</b> HP</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.subheader("🤖 Estado del modelo")

        m1, m2 = st.columns(2)

        with m1:
            st.markdown('<div class="result-number">{:.2f}</div>'.format(accuracy), unsafe_allow_html=True)
            st.write("Accuracy")

        with m2:
            st.markdown('<div class="result-number">{:.2f}</div>'.format(f1), unsafe_allow_html=True)
            st.write("F1 Score")

        st.progress(float(accuracy))

        st.write("""
        Modelo utilizado: **Árbol de Decisión**  
        Variable objetivo: **Vehicle type**
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Vehicle Analytics AI | Proyecto Final | Carlos Alfredo González Hernández | Diplomado IA 2026
    </div>
    """, unsafe_allow_html=True)

else:
    st.title(menu)
    st.info("Esta sección se integrará en la siguiente etapa del desarrollo profesional.")
