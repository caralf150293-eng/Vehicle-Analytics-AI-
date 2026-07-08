# 🚗 Vehicle Analytics AI

Dashboard interactivo para análisis, visualización y clasificación de vehículos mediante técnicas de Ciencia de Datos y Machine Learning.

---

# 📌 Descripción

Vehicle Analytics AI es una aplicación desarrollada en **Python** y **Streamlit** que permite analizar un conjunto de datos de una agencia automotriz mediante técnicas de análisis exploratorio, visualización interactiva y aprendizaje automático.

La aplicación integra un panel ejecutivo con indicadores clave, gráficos dinámicos y un modelo predictivo basado en **Árbol de Decisión**, facilitando el análisis de información y la clasificación de vehículos.

---

# ✨ Funcionalidades

- 📊 Dashboard Ejecutivo
- 📈 Visualizaciones Interactivas
- 🔥 Matriz de Correlación
- 🤖 Modelo de Machine Learning
- 🎯 Predicción de vehículos
- 📄 Reporte Ejecutivo

---

# 🖥 Vista principal

![Vista principal](images/home.png)

La pantalla de inicio reúne la identidad visual de la aplicación, indicadores principales y el acceso a todos los módulos del sistema.

---

# 📊 Dashboard Ejecutivo

![Dashboard Ejecutivo](images/dashboard.png)

Permite visualizar los principales indicadores del conjunto de datos mediante KPIs y gráficos interactivos, facilitando la interpretación de la información.

---

# 🔥 Matriz de Correlación

![Matriz de Correlación](images/correlacion.png)

Presenta la relación entre las variables numéricas del conjunto de datos, permitiendo identificar asociaciones importantes para el análisis y el entrenamiento del modelo.

---

# 🤖 Modelo Predictivo

![Modelo Predictivo](images/modelo.png)

Se implementó un modelo de **Árbol de Decisión** para clasificar el tipo de vehículo utilizando variables numéricas como:

- Precio
- Tamaño del motor
- Horsepower
- Wheelbase
- Width

El modelo fue evaluado mediante métricas de desempeño como Accuracy, Precision, Recall y F1 Score.

---

# 🎯 Predicción de Vehículos

![Predicción](images/prediccion.png)

El usuario puede ingresar las características de un nuevo vehículo y obtener una predicción automática utilizando el modelo previamente entrenado.

---

# 🛠 Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📂 Estructura del proyecto

```text
Vehicle-Analytics-AI
│
├── data
├── images
├── app_pro_v2.py
├── Desarrollo_Streamlit.ipynb
├── README.md
└── requirements.txt
```

---

# 🚀 Próximas mejoras

- Publicación en Streamlit Community Cloud.
- Exportación de reportes PDF.
- Exportación de resultados a Excel.
- Integración con base de datos.
- Sistema de autenticación de usuarios.

---

# 👤 Autor

**Carlos Alfredo González Hernández**

Aplicación desarrollada con Python, Streamlit y Scikit-learn como proyecto de análisis de datos y Machine Learning.

# 🌐 Aplicación en línea

Puedes probar Vehicle Analytics AI aquí:

https://tu-app.streamlit.app
