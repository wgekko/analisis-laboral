IMPORTANTE 
deben crear una carpeta con un punto delante  .streamtlit (es la base de los colores y tipo de letras del proyecto)
que contenga un arhivo llamado 
config.toml
con estos datos dentro
[server]
enableStaticServing = true

[[theme.fontFaces]]
family = "Inter"
url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"

[theme]
primaryColor = "#FF8C00"
backgroundColor = "#0D1B2A"
secondaryBackgroundColor = "#1B263B"
textColor = "#FFA500"
linkColor = "#FFA500"
borderColor = "#CCCCCC"
showWidgetBorder = true
baseRadius = "0.5rem"
buttonRadius = "0.5rem"
font = "Inter"
headingFontWeights = [600, 500]
headingFontSizes = ["2.5rem", "1.8rem"]
codeFont = "Courier New"
codeFontSize = "0.75rem"
codeBackgroundColor = "#112B3C"
showSidebarBorder = false
chartCategoricalColors = [
  "#FF8C00",  # Orange oscuro
  "#FFA500",  # Naranja clásico
  "#FFD700",  # Mostaza / dorado
  "#E1C16E",  # Mostaza claro
  "#C8E25D",  # Lima suave
  "#A8D08D",  # Verde pastel
  "#7AC36A",  # Verde hoja
  "#4CAF50",  # Verde medio
  "#40C4FF",  # Celeste vibrante
  "#00B0F0",  # Celeste profesional
  "#3399FF",  # Celeste más oscuro
  "#1E88E5",  # Azul Francia
  "#1976D2",  # Azul fuerte
  "#1565C0",  # Azul oscuro
  "#0D47A1"   # Azul muy profundo
]

chartCategoricalColors1 = [
  "#FF8C00",
  "#FFA500",
  "#FFB347",
  "#FFD580",
  "#FFA07A",
  "#FF7F50",
  "#FF6F00",
  "#CC7000",
  "#FFC107",
  "#FFDD57",
  "#E67E22",
  "#D35400",
  "#F39C12",
  "#E67E22",
  "#F4A261"
]

[theme.sidebar]
backgroundColor = "#1E3A5F"
secondaryBackgroundColor = "#1B263B"
headingFontSizes = ["1.6rem", "1.4rem", "1.2rem"]
dataframeHeaderBackgroundColor = "#1A2A40"

----------------------------------------------------------------------------------------------------

Análisis y Predicción Socioeconómica - Tasa/Población Ocupada, Desocupación y Subocupación
 Descripción General

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


video demo 

https://github.com/user-attachments/assets/c8bdc556-9081-4074-b94a-049f0bcb2c38




Este proyecto se distribuye bajo licencia MIT, lo que permite su uso, modificación y distribución con atribución.

