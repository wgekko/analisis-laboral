🧠 Análisis y Predicción Socioeconómica - Tasa/Población Ocupada, Desocupación y Subocupación
📊 Descripción General

Este proyecto permite analizar, modelar y predecir series temporales socioeconómicas relacionadas con el mercado laboral argentino (u otras regiones), utilizando datos trimestrales de empleo, desocupación, subocupación y trabajo informal.

Los módulos están implementados con Streamlit para una interfaz interactiva, e integran técnicas de estadística, machine learning y análisis fractal para comprender la estructura dinámica de las series.

Actualmente incluye:

ocupacion_app.py → Análisis y predicción de tasa/población ocupada

desocupacion_app.py → Análisis y predicción de tasa de desocupación

subocupacion_app.py → Análisis y predicción de tasa de subocupación

informalidad_app.py → Análisis y predicción de trabajo informal

Cada módulo sigue la misma mecánica de carga, exploración, descomposición temporal, pronóstico y análisis fractal.

⚙️ Características principales

Carga dinámica de datos desde archivos Excel (/data/<tema>/*.xlsx), con encabezado en la fila 4.

Visualización interactiva de series temporales mediante Plotly.

Descomposición STL y pronóstico ARIMA / SARIMAX / pmdarima.

Análisis fractal y de persistencia mediante:

Exponente de Hurst

Detrended Fluctuation Analysis (DFA)

Fractal Dimension (Box-counting)

Interpretación neosimbólica (ARAD) para caracterizar dinámicas sociales:

GUERRERO (acción), INTELECTUAL (reflexión), LOGRERO (acumulación), MENESTEROSO (crisis)

Modelos explicativos (Machine Learning):

LinearRegression, RandomForestRegressor, XGBoost (opcional)

Clasificación opcional (árboles y regresión logística)

Modelos híbridos ARIMA + ML sobre residuos.

Predicción LSTM (PyTorch) opcional para series extendidas.

Análisis de “espines fractales” (validación de grados enteros simbólicos).


Instalación y ejecución
- Clonar el repositorio
git clone https://github.com/usuario/analisis-laboral.git
cd analisis-laboral

- Crear un entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt

- Estructurar los datos

Coloca los archivos Excel dentro de:

data/ocupacion/
data/desocupacion/
data/subocupacion/
data/informalidad/


Cada archivo debe tener el encabezado en la fila 4 (índice 3).

- Ejecutar la aplicación
streamlit run ocupacion_app.py


Luego abre el enlace que Streamlit muestra (por defecto: http://localhost:8501
).

- Ejemplo de uso

Cargar el archivo Excel de la carpeta correspondiente.

Seleccionar la categoría (por ejemplo, “Varones 25–45 años”).

Presionar “Analizar categoría seleccionada”.

Explorar:

Gráfico de serie temporal

Descomposición y pronóstico

Análisis fractal y neosimbólico

Modelos de predicción y clasificación

Exportación a CSV

- Dependencias principales

Las librerías utilizadas están especificadas en requirements.txt, e incluyen:

streamlit

pandas, numpy, plotly

statsmodels, scikit-learn

pmdarima, xgboost, torch (opcionales)

🧬 Conceptos teóricos implementados

Hurst Exponent (H): mide persistencia o aleatoriedad temporal.

DFA (Detrended Fluctuation Analysis): detecta auto-similitud fractal.

Fractal Box Dimension: mide complejidad geométrica de la serie.

ARAD (Arquetipos Dinámicos): clasificación simbólica según tendencia y persistencia.

Espines fractales: interpretación simbólica de fases fractales según grados enteros.

Autores y créditos

Desarrollado por [Walter Gomez]
Inspirado en metodologías de análisis fractal aplicado a series socioeconómicas..

Licencia

Este proyecto se distribuye bajo licencia MIT, lo que permite su uso, modificación y distribución con atribución.