import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


# Dependencias opcionales
try:
    import pmdarima as pm
    PM_AVAILABLE = True
except Exception:
    PM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


st.set_page_config(layout='wide', page_title='Análisis y Predicción - Tasa/Población Ocupada', page_icon=":material/analytics:",initial_sidebar_state="collapsed" )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# ------------------ Funciones auxiliares ------------------

def to_quarter_timestamp(label):
    """
    Convierte etiquetas tipo '3° Trim 03' -> Timestamp (inicio del trimestre)
    """
    txt = str(label)
    txt = re.sub(r"[^0-9A-Za-z° ]", "", txt)
    txt = txt.replace('Trim', 'T').replace('trim', 'T')
    m = re.search(r"([1-4])\D*([0-9]{2,4})", txt)
    if m:
        q = int(m.group(1))
        yr = m.group(2)
        if len(yr) == 2:
            yr = int('20' + yr)
        else:
            yr = int(yr)
        return pd.Period(f"{yr}Q{q}").to_timestamp()
    return pd.NaT

def hurst_exponent(ts):
    ts = np.array(pd.Series(ts).dropna())
    N = len(ts)
    if N < 20:
        return np.nan
    Y = np.cumsum(ts - np.mean(ts))
    R, S = [], []
    for n in range(10, N):
        Rn = np.max(Y[:n]) - np.min(Y[:n])
        Sn = np.std(ts[:n])
        if Sn == 0:
            continue
        R.append(Rn); S.append(Sn)
    R = np.array(R); S = np.array(S)
    mask = (S > 0) & (R > 0)
    if mask.sum() < 2:
        return np.nan
    import scipy.stats as stats
    lr = stats.linregress(np.log(np.arange(10, 10 + mask.sum())), np.log(R[mask] / S[mask]))
    return lr.slope

def fractal_dimension_boxcount(Z):
    Z = np.array(Z)
    if len(Z) < 8:
        return np.nan
    x = np.linspace(0,1,len(Z))
    y = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-12)
    size = 64
    grid = np.zeros((size,size), dtype=bool)
    xi = (x*(size-1)).astype(int)
    yi = (y*(size-1)).astype(int)
    grid[yi, xi] = True
    counts = []
    sizes = []
    k = 1
    while k <= size:
        cnt = 0
        for i in range(0,size,k):
            for j in range(0,size,k):
                if grid[i:i+k, j:j+k].any():
                    cnt += 1
        counts.append(cnt); sizes.append(size/k)
        k *= 2
    counts = np.array(counts); sizes = np.array(sizes)
    mask = counts > 0
    if mask.sum() < 2:
        return np.nan
    import scipy.stats as stats
    lr = stats.linregress(np.log(sizes[mask]), np.log(counts[mask]))
    return -lr.slope

def dfa(ts):
    x = np.array(pd.Series(ts).dropna()) - np.nanmean(ts)
    N = len(x)
    if N < 10:
        return np.nan
    y = np.cumsum(x)
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(N/4), num=10)).astype(int))
    F = []
    for s in scales:
        nseg = N // s
        if nseg < 2:
            continue
        rms = []
        for v in range(nseg):
            seg = y[v*s:(v+1)*s]
            t = np.arange(len(seg))
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        if len(rms):
            F.append(np.mean(rms))
    if len(F) < 2:
        return np.nan
    import scipy.stats as stats
    lr = stats.linregress(np.log(scales[:len(F)]), np.log(F))
    return lr.slope

PROB_FRACTAL = 2 / np.pi  # ≈ 0.6366

# ---------- ARAD / Socioeconomía neosimbólica ----------

def arad_state(ts):
    ts = pd.Series(ts).dropna()
    if len(ts) < 4:
        return 'Indeterminado', None
    recent = ts.iloc[-8:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    hurst = hurst_exponent(ts)
    if slope > 0 and (not np.isnan(hurst)) and hurst >= 0.6:
        return 'GUERRERO (Action)', {'slope': slope, 'hurst': hurst}
    if slope > 0 and (np.isnan(hurst) or hurst < 0.6):
        return 'INTELECTUAL (Reflection)', {'slope': slope, 'hurst': hurst}
    if slope <= 0 and (not np.isnan(hurst)) and hurst >= 0.6:
        return 'LOGRERO (Accumulation)', {'slope': slope, 'hurst': hurst}
    return 'MENESTEROSO (Desperation)', {'slope': slope, 'hurst': hurst}

def socioeconomia_neosimbolica_interpretacion(hurst, dfa_alpha, tendencia):
    if np.isnan(hurst):
        return "No se puede determinar el comportamiento fractal."
    if hurst > 0.6 and tendencia > 0:
        return "Sociedad activa con predominio GUERRERO — acción y crecimiento."
    elif hurst > 0.6 and tendencia < 0:
        return "Sociedad LOGRERA — acumulación y conservación de capital."
    elif hurst < 0.5 and tendencia > 0:
        return "Sociedad INTELECTUAL — reflexión y transición expansiva."
    else:
        return "Sociedad MENESTEROSA — desesperación, crisis o contracción."

# ---------- Espines fractales (validación entera de grados según tu especificación) ----------
ALLOWED_SPIN_FRACS = [0.5, 1.5, 2.5, 4.5, 7.5, 22.5]  # 1/2,3/2,5/2,9/2,15/2,45/2

def spin_degree_from_frac(frac):
    if frac == 0:
        return np.inf
    return 360.0 / frac

def is_integer_degree(deg, tol=1e-8):
    return abs(deg - round(deg)) < tol

def spin_fractal_analysis(hurst, dfa_alpha, box_dim):
    if np.isnan(hurst) or np.isnan(dfa_alpha) or np.isnan(box_dim):
        return "Datos insuficientes para análisis de espín fractal.", None

    # heurística (ajustable)
    chosen = None
    desc = None
    if 0.50 <= hurst <= 0.65 and 0.8 <= dfa_alpha <= 1.1 and box_dim <= 1.25:
        chosen = 0.5
        desc = "Espín 1/2: trayectoria canónica y coherente — persistencia/moderado orden."
    elif hurst > 0.65 and dfa_alpha >= 0.9:
        chosen = 1.5
        desc = "Espín 3/2: cambio de fase / reconfiguración estructural."
    elif dfa_alpha > 1.25 or box_dim > 1.3:
        chosen = 2.5
        desc = "Espín 5/2: alta turbulencia fractal — comportamiento complejo."
    elif hurst < 0.45:
        chosen = 4.5
        desc = "Espín 9/2: fractal degenerado — pérdida de correlación."
    elif hurst < 0.40 and box_dim > 1.4:
        chosen = 7.5
        desc = "Espín 15/2: ruptura estructural — señales de crisis."
    else:
        chosen = 22.5
        desc = "Espín 45/2: caos determinista aparente o recorrido posicional imposible."

    deg = spin_degree_from_frac(chosen)
    valid = is_integer_degree(deg)
    # fallback: intentar 1/2 si el elegido no cumple entero
    if not valid:
        deg05 = spin_degree_from_frac(0.5)
        if is_integer_degree(deg05):
            chosen = 0.5
            deg = deg05
            desc += " (reasignado a 1/2 por condición posicional entera)."

    info = {
        'spin_frac': chosen,
        'spin_label': f"{int(chosen*2)}/{2}" if chosen != 22.5 else "45/2",
        'degrees': round(deg,6),
        'valid_degree_integer': is_integer_degree(deg),
        'hurst': hurst,
        'dfa': dfa_alpha,
        'box_dim': box_dim
    }
    return desc, info

# ------------------ Carga de datos (lee header en fila 4) ------------------
@st.cache_data
def load_occupacion_data(path_or_buf):
    # lee encabezado en fila 4 (index=3)
    if isinstance(path_or_buf, str):
        df_raw = pd.read_excel(path_or_buf, header=3)
    else:
        df_raw = pd.read_excel(path_or_buf, header=3)

    # renombra columna categoría si tiene otro nombre parecido
    if 'Categoría' not in df_raw.columns:
        posibles = [c for c in df_raw.columns if 'categ' in str(c).lower()]
        if posibles:
            df_raw.rename(columns={posibles[0]: 'Categoría'}, inplace=True)

    df_raw.dropna(axis=1, how='all', inplace=True)
    cols_data = [c for c in df_raw.columns if c != 'Categoría']

    # convertir columnas a numérico cuando sea posible
    for c in cols_data:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

    numeric_sum = df_raw[cols_data].fillna(0).abs().sum(axis=0)
    valid_cols = numeric_sum[numeric_sum != 0].index.tolist()
    df_keep = df_raw[['Categoría'] + valid_cols]

    # melt
    df_long = df_keep.melt(id_vars='Categoría', var_name='Trimestre', value_name='Valor')
    df_long['Trimestre_clean'] = df_long['Trimestre'].astype(str).str.replace(r"[^0-9A-Za-z° ]", "", regex=True)
    df_long['Fecha'] = df_long['Trimestre_clean'].apply(to_quarter_timestamp)

    df_long = df_long.dropna(subset=['Fecha', 'Valor'])
    df_long['Valor'] = pd.to_numeric(df_long['Valor'], errors='coerce')
    df_long = df_long[df_long['Valor'] > 0]
    df_long['Categoría'] = df_long['Categoría'].astype(str).str.strip()
    return df_long

# ------------------ PyTorch LSTM (opcional) ------------------
if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

# ------------------ Interfaz Streamlit ------------------
st.title(":material/badge: Análisis y Predicción - Población/Tasa Ocupada (2003-2025)")

st.info("Datos de Población Ocupada- según características socio-economicas")
st.info("Datos de Población Ocupada- según características socio-economicas excluyendo beneficios planes empleo")
st.info("Datos de tasa de empleo según características socio-economicas")
st.markdown("Carga el Excel. Selecciona categoría y presiona **Analizar categoría seleccionada**.")

# ------------------------------------------------------------
# CARGA AUTOMÁTICA DEL EXCEL
# ------------------------------------------------------------
ruta_base = os.path.join(os.getcwd(), "data", "ocupacion")
if not os.path.exists(ruta_base):
    st.error("La carpeta 'data/ocupacion' no existe. Crea esa ruta y coloca allí los archivos Excel.")
    st.stop()

archivos_excel = [f for f in os.listdir(ruta_base) if f.endswith((".xlsx", ".xls"))]

if not archivos_excel:
    st.error("No se encontraron archivos Excel en la carpeta 'data/ocupacion'.")
    st.stop()

archivo_sel = st.selectbox("Selecciona el archivo de datos:", archivos_excel)
ruta_archivo = os.path.join(ruta_base, archivo_sel)

try:
    df_raw = pd.read_excel(ruta_archivo, header=3)
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
    st.stop()



#uploaded = st.file_uploader("Sube el Excel (fila 4 = encabezado)", type=["xlsx", "xls"])
agg_choice = st.sidebar.selectbox("Agregación para duplicados Categoria+Fecha", ["mean", "sum"])
min_obs = st.sidebar.slider("Mínimo observaciones para análisis temporal", 8, 60, 12)

if ruta_archivo:
    st.audio("audio/Harry_Gregson-Williams_-_Operation_Dinner_Out_SPY_GAME_OST_(mp3.pm).mp3", format="audio/mp3",loop=True, autoplay=True)
    df_long = load_occupacion_data(ruta_archivo)
    st.success(f"Datos cargados: {df_long.shape[0]} filas válidas")
    st.dataframe(df_long.head(12))
    

    # detectar duplicados combinados (info al usuario)
    n_dups = df_long.duplicated(subset=['Categoría','Fecha'], keep=False).sum()
    if n_dups:
        st.warning(f"Se detectaron {n_dups} filas duplicadas en combinaciones Categoría+Fecha. Se aplicará agregación: {agg_choice}")

    categorias = sorted(df_long['Categoría'].unique())
    categoria_sel = st.sidebar.selectbox("Selecciona categoría para analizar", categorias)
    

    if st.sidebar.button("Analizar categoría seleccionada"):
        # serie de la categoría seleccionada
        df_cat = df_long[df_long['Categoría'] == categoria_sel].copy()
        ts = df_cat.groupby('Fecha')['Valor'].sum().sort_index()

        # time series plot
        st.subheader(f"Serie temporal — {categoria_sel}")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers', name=categoria_sel))
        fig_ts.update_layout(title=f"Serie temporal - {categoria_sel}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
        st.plotly_chart(fig_ts, use_container_width=True)

        # STL / ARIMA / fractales
        if len(ts) >= min_obs:
            st.subheader("Descomposición y pronóstico")
            try:
                from statsmodels.tsa.seasonal import STL
                stl = STL(ts, period=4, robust=True).fit()
                comp = pd.DataFrame({'trend': stl.trend, 'seasonal': stl.seasonal, 'resid': stl.resid})
                figc = go.Figure()
                figc.add_trace(go.Scatter(x=comp.index, y=comp['trend'], name='Trend'))
                figc.add_trace(go.Scatter(x=comp.index, y=comp['seasonal'], name='Seasonal'))
                figc.add_trace(go.Scatter(x=comp.index, y=comp['resid'], name='Resid'))
                figc.update_layout(title='STL decomposition', template='plotly_white')
                st.plotly_chart(figc, use_container_width=True)
            except Exception as e:
                st.warning("STL error: " + str(e))

            # ARIMA / SARIMAX
            st.subheader("Pronóstico ARIMA / SARIMAX (8 períodos)")
            period = 4
            try:
                if PM_AVAILABLE:
                    model = pm.auto_arima(ts, seasonal=True, m=period, suppress_warnings=True)
                    fc = model.predict(n_periods=8)
                    idx = pd.period_range(ts.index[-1].to_period('Q')+1, periods=8, freq='Q').to_timestamp()
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Observed'))
                    figf.add_trace(go.Scatter(x=idx, y=fc, name='ARIMA forecast'))
                    st.plotly_chart(figf, use_container_width=True)
                else:
                    sar = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,period),
                                                    enforce_stationarity=False, enforce_invertibility=False)
                    res = sar.fit(disp=False)
                    fc = res.get_forecast(8).predicted_mean
                    figf = go.Figure()
                    figf.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Observed'))
                    figf.add_trace(go.Scatter(x=fc.index, y=fc.values, name='SARIMAX forecast'))
                    st.plotly_chart(figf, use_container_width=True)
            except Exception as e:
                st.warning("ARIMA/SARIMAX error: " + str(e))

            # fractal measures
            hurst = hurst_exponent(ts)
            dfa_alpha = dfa(ts)
            box_dim = fractal_dimension_boxcount(ts.values)
            tendencia = np.polyfit(np.arange(len(ts)), ts.values, 1)[0]
            st.subheader("Análisis fractal y Neosimbólico")
            st.write({'Hurst': hurst, 'DFA alpha': dfa_alpha, 'Box-counting dim': box_dim, 'Probabilidad fractal (2/pi)': PROB_FRACTAL})
            st.info(socioeconomia_neosimbolica_interpretacion(hurst, dfa_alpha, tendencia))

            state, info = arad_state(ts)
            st.write(f"**Estado ARAD:** {state}")
            st.write(info)

            # espines fractales (validación entera grados)
            st.subheader("Espines fractales (Diagrama de Fases) :material/shape_line:")
            desc_spin, s_info = spin_fractal_analysis(hurst, dfa_alpha, box_dim)
            st.write(desc_spin)
            if s_info:
                st.write(f"Espín (frac): {s_info['spin_frac']} — etiqueta: {s_info['spin_label']}")
                st.write(f"Grados: {s_info['degrees']} — Entero válido: {s_info['valid_degree_integer']}")
                st.write(f"Métricas (Hurst, DFA, BoxDim): {s_info['hurst']:.3f}, {s_info['dfa']:.3f}, {s_info['box_dim']:.3f}")

        else:
            st.warning(f"Serie con {len(ts)} observaciones: se requieren al menos {min_obs} para análisis temporal robusto.")

        # ---------------- ML explicativo ----------------
        st.subheader("Modelos explicativos (Regresión / RandomForest / XGBoost opcional)")

        # agregación previa para evitar duplicados en pivot
        df_long_agg = df_long.groupby(['Categoría','Fecha'], as_index=False)['Valor'].agg(agg_choice)
        df_pivot = df_long_agg.pivot(index='Categoría', columns='Fecha', values='Valor').fillna(0)
        # normalizar columnas duplicadas por si acaso
        df_pivot = df_pivot.loc[:, ~df_pivot.columns.duplicated()]

        if df_pivot.shape[1] >= 4:
            features = df_pivot.columns[:-1]
            target = df_pivot.columns[-1]
            X = df_pivot[features]
            y = df_pivot[target]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
            lr = LinearRegression().fit(Xtr_s, ytr)
            st.write("Linear R2:", round(r2_score(yte, lr.predict(Xte_s)),3))
            rf = RandomForestRegressor(n_estimators=150, random_state=42).fit(Xtr, ytr)
            st.write("RandomForest R2:", round(r2_score(yte, rf.predict(Xte)),3))
            st.dataframe(pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(10))
            if XGB_AVAILABLE:
                try:
                    dtrain = xgb.DMatrix(Xtr, label=ytr); dtest = xgb.DMatrix(Xte, label=yte)
                    params = {'objective':'reg:squarederror', 'max_depth':4, 'eta':0.1}
                    bst = xgb.train(params, dtrain, num_boost_round=80)
                    st.write("XGBoost R2:", round(r2_score(yte, bst.predict(dtest)),3))
                except Exception as e:
                    st.warning("XGBoost fallo: " + str(e))
        else:
            st.info("No hay suficientes columnas temporales (>=4) para entrenar modelos explicativos.")

        # ---------------- Clasificación (opcional) ----------------
        st.subheader("Clasificación (opcional)")
        cat_cands = [c for c in df_long.columns if df_long[c].dtype == object and df_long[c].nunique() <= 10]
        if cat_cands:
            tcol = st.selectbox("Selecciona target categórico", cat_cands)
            if tcol:
                feats = df_long.select_dtypes(include=[np.number]).columns.tolist()
                if len(feats) >= 2:
                    data_clf = df_long[feats + [tcol]].dropna()
                    X = data_clf[feats]; y = LabelEncoder().fit_transform(data_clf[tcol])
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                    clf = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)
                    st.write("DecisionTree acc:", round(accuracy_score(yte, clf.predict(Xte)),3))
                    log = LogisticRegression(max_iter=500).fit(Xtr, ytr)
                    st.write("Logistic acc:", round(accuracy_score(yte, log.predict(Xte)),3))
                else:
                    st.warning("No hay features numéricas suficientes para clasificación.")
        else:
            st.info("No se detectaron columnas categóricas cortas para clasificación.")

        # ---------------- Híbrido ARIMA + RF en residuos ----------------
        st.subheader("Híbrido ARIMA + RF sobre residuos")
        try:
            if len(ts) >= min_obs:
                sar = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,4),
                                                enforce_stationarity=False, enforce_invertibility=False)
                res = sar.fit(disp=False)
                resid = ts - res.fittedvalues
                df_ag = df_long.groupby('Fecha').mean(numeric_only=True)
                df_ml = df_ag.join(resid.rename('resid'), how='inner').dropna()
                if df_ml.shape[0] >= 10:
                    X = df_ml.drop(columns=['resid']); y = df_ml['resid']
                    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2, random_state=42)
                    rf2 = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr,ytr)
                    st.write("RF sobre residuos R2:", round(r2_score(yte, rf2.predict(Xte)),3))
        except Exception as e:
            st.warning("Híbrido ARIMA+ML fallo: " + str(e))

        # ---------------- LSTM PyTorch forecasting ----------------
        st.subheader("LSTM (PyTorch) forecasting")
        if not TORCH_AVAILABLE:
            st.info("PyTorch no instalado. Para LSTM instala torch (ej: pip install torch).")
        else:
            if len(ts) >= max(min_obs,12):
                try:
                    vals = ts.values.reshape(-1,1).astype(np.float32)
                    mean = vals.mean(); std = vals.std() if vals.std()>0 else 1.0
                    vals_s = (vals - mean)/std
                    WINDOW = min(8, max(4, len(vals_s)//4))
                    X=[]; Y=[]
                    for i in range(len(vals_s)-WINDOW):
                        X.append(vals_s[i:i+WINDOW]); Y.append(vals_s[i+WINDOW])
                    X = np.stack(X); Y = np.stack(Y)
                    split = int(0.8*len(X))
                    Xtr, Xte = X[:split], X[split:]; Ytr, Yte = Y[:split], Y[split:]
                    train_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
                    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
                    model = LSTMNet(input_size=1, hidden_size=32, num_layers=1)
                    opt = torch.optim.Adam(model.parameters(), lr=0.01); loss_fn = nn.MSELoss()
                    model.train()
                    EPOCHS = 40
                    pbar = st.progress(0)
                    for e in range(EPOCHS):
                        for xb, yb in train_loader:
                            opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
                        pbar.progress(int((e+1)/EPOCHS*100))
                    model.eval()
                    last = vals_s[-WINDOW:].reshape(1,WINDOW,1).astype(np.float32); inp = torch.from_numpy(last).float()
                    preds = []
                    for _ in range(8):
                        with torch.no_grad():
                            p = model(inp).numpy().reshape(-1)
                        preds.append(p[0])
                        arr = inp.numpy(); arr = np.roll(arr, -1); arr[0,-1,0] = p[0]; inp = torch.from_numpy(arr).float()
                    preds_inv = (np.array(preds).reshape(-1,1)*std) + mean
                    idx = pd.period_range(ts.index[-1].to_period('Q')+1, periods=8, freq='Q').to_timestamp()
                    fig = go.Figure(); fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Observed'))
                    fig.add_trace(go.Scatter(x=idx, y=preds_inv.flatten(), name='LSTM (PyTorch)'))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("LSTM error: " + str(e))
            else:
                st.warning("Serie demasiado corta para LSTM (se recomiendan >=12 observaciones).")

        # ---------------- Export pivot CSV ----------------
        if st.button("Descargar pivot CSV (agregado)"):
            csv = df_long_agg = df_long.groupby(['Categoría','Fecha'], as_index=False)['Valor'].agg(agg_choice)
            csv = csv.pivot(index='Categoría', columns='Fecha', values='Valor').fillna(0).to_csv()
            st.download_button("Descargar CSV", csv, file_name='ocupacion_pivot.csv', mime='text/csv')

        st.success("Pipeline completo finalizado.")

else:
    st.info("Sube el Excel (fila 4 = encabezado) y luego selecciona categoría para analizar.")
