import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Predicción de Siniestros",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    model = load_model("modelo_siniestro.h5")
    scaler = joblib.load("scaler.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    return model, scaler, columnas

modelo, scaler, columnas_modelo = cargar_modelo()

with st.sidebar:
    st.image("logo_confianza.png", width=160)
    st.image("logo_javeriana.png", width=140)
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 14px; text-align: left;'>
        <p><Modelo de recomendación
        que permita disminuir la siniestralidad en el otorgamiento de pólizas de cumplimiento de los clientes con negocios suscritos en Seguros Confianza S.A</p>
        <br>
        <p><strong>Presentado por:</strong><br>
        Yulieth Danitza Aguillón Ortega<br>
        Diana Cristina Benavides Fonseca<br>
        Edgar Antonio Cruz Martínez<br>
        Luis Alejandro Garzón Ramírez</p>
        <br>
        <p><strong>Trabajo de grado aplicado</strong><br>
        para optar al título de Magíster en Analítica para la Inteligencia de Negocios</p>
        <br>
        <p><strong>Asesorado por:</strong><br>
        Cristian Camilo Tirado Cifuentes</p>
        <br>
        <p>Pontificia Universidad Javeriana<br>
        Facultad de Ingeniería<br>
        Maestría en Analítica para la Inteligencia de Negocios<br>
        Bogotá D.C · 2025</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <h2 style='text-align: left;'>🧠 Predicción de Siniestros</h2>
    <h5 style='text-align: left; color: gray;'>Herramienta predictiva institucional de Seguros Confianza y la Pontificia Universidad Javeriana</h5>
    <hr>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    vaseg = st.number_input("💰 Valor asegurado", 0.0, 1e9, 5_000_000.0, step=100_000.0)
    cambio = st.number_input("🔁 Cambio riesgo", 0.0, 10.0, 0.0)
    amparo_con = st.radio("✅ ¿Incluye Amparo CON?", ["No", "Sí"], horizontal=True)
    ciuu = st.selectbox("🏭 División CIUU", [chr(x) for x in range(ord("A"), ord("U") + 1)])

with col2:
    porcom = st.slider("📊 Comisión Intermediario (%)", 0.0, 100.0, 5.0)
    vigencia = st.slider("📅 Tiempo vigencia póliza (meses)", 1, 60, 12)
    tipo_tomador = st.selectbox("👤 Tipo de tomador", ["UNICO", "CONSORCIO / UT"])
    facultativo = st.radio("📃 ¿Es facultativo?", ["No", "Sí"], horizontal=True)

with col3:
    riesgo = st.slider("⚠️ Calificación riesgo última", 0.0, 5.0, 3.0)
    cat_prod = st.selectbox("🏷️ Categoría de producto", [
        "CAUCIONES JUDICIALES", "DISPOSICIONES LEGALES", "ECOPETROL",
        "EMPR. SERVICIOS PUBLICOS", "GARANTIA UNICA", "PARTICULAR",
        "PARTICULAR M. EXTRANJERA", "Otros"
    ])
    tipcoa = st.selectbox("🔢 Tipo de operación (TIPCOA)", ["1", "2", "3"])

col_button, col_result = st.columns([1, 2])
with col_button:
    calcular = st.button("🚀 Calcular probabilidad de siniestro", use_container_width=True)

if calcular:
    datos = {
        "VASEG_GROSS": vaseg,
        "PORCOMINTER": porcom,
        "CALIF_RIESGO_ULTIMA": riesgo,
        "CAMBIO_RIESGO": cambio,
        "tiempovigencia_poliza": vigencia,
        "FACULTATIVO_PSEUDO": 1 if facultativo == "Sí" else 0,
        "AMPARO_CON": 1 if amparo_con == "Sí" else 0
    }

    # Asignación de TIPCOA
    for i in ["1", "2", "3"]:
        datos[f"TIPCOA_{i}"] = 1 if tipcoa == i else 0

    for col in columnas_modelo:
        if col not in datos:
            datos[col] = 0

    datos[f"Cat_prod_{cat_prod}"] = 1
    datos[f"DIVISION_CIUU_{ciuu}"] = 1
    datos["tipotomador_UNICO"] = 1 if tipo_tomador == "UNICO" else 0
    datos["tipotomador_CONSORCIO / UT"] = 1 if tipo_tomador != "UNICO" else 0

    X = pd.DataFrame([datos])[columnas_modelo]
    X_scaled = scaler.transform(X)
    prob = float(modelo.predict(X_scaled)[0][0])

    # --- SHAP robust analysis ---
    import shap
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        # Usar KernelExplainer para compatibilidad general
        X_sample = X.sample(n=1, random_state=0)
        explainer = shap.KernelExplainer(modelo.predict, X_sample)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

        # Para modelos binarios, shap_values puede ser una lista con un solo elemento
        if isinstance(shap_values, list):
            shap_vals_row = np.array(shap_values[0][0])
            base_val = float(np.array(explainer.expected_value).flatten()[0])
        else:
            shap_vals_row = np.array(shap_values[0])
            base_val = float(np.array(explainer.expected_value).flatten()[0])

        # Debug: muestra shapes antes de construir el Explanation
        #st.write('shap_vals_row.shape:', shap_vals_row.shape)
        #st.write('X_scaled.flatten().shape:', X_scaled.flatten().shape)
        #st.write('base_val:', base_val)
        #st.write('features:', list(X.columns))

        exp = shap.Explanation(
            values=shap_vals_row.flatten(),
            base_values=base_val,
            data=X_scaled.flatten(),
            feature_names=list(X.columns)
        )

        st.markdown("<h5>🔍 Análisis SHAP (explicabilidad del modelo)</h5>", unsafe_allow_html=True)
        plt.gcf().set_size_inches(10, 6)
        shap.plots.bar(exp, max_display=10, show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"No se pudo generar el gráfico SHAP: {e}")

    with col_result:
        st.subheader("📊 Resultado de predicción")
        st.metric("🔮 Probabilidad estimada de siniestro", f"{prob:.2%}")
        st.progress(min(max(prob, 0.0), 1.0))

        color = "red" if prob >= 0.6 else "orange" if prob >= 0.4 else "green"
        riesgo_txt = "ALTO" if prob >= 0.6 else "MODERADO" if prob >= 0.4 else "BAJO"

        st.markdown(f"""
        <div style='background-color:{color};padding:1rem;border-radius:8px;text-align:center'>
            <h3 style='color:white;'>🛑 Nivel de Riesgo: {riesgo_txt}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><h5>📄 Variables procesadas:</h5>", unsafe_allow_html=True)
        st.dataframe(X.round(2), use_container_width=True)

# === SECCIÓN: Ejemplos seleccionables con explicación SHAP ===
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

st.header("🔎 Ejemplos reales de test y su explicación SHAP")

# Rutas y carga del test (ajusta si ya tienes cargado df_test)
ruta_base = "data/"
df_test = pd.read_csv(os.path.join(ruta_base, "base_con_ciuu_test.csv"))

X_test = df_test.drop(columns=["siniestro"])
y_test = df_test["siniestro"]

# Escala
X_test_scaled = scaler.transform(X_test)
probs = modelo.predict(X_test_scaled).flatten()
df_resultados = X_test.copy()
df_resultados['probabilidad'] = probs
df_resultados['real'] = y_test.values

# Top 3 sí y top 3 no siniestro
top_si = df_resultados.sort_values('probabilidad', ascending=False).head(3)
top_no = df_resultados.sort_values('probabilidad', ascending=True).head(3)
ejemplos = pd.concat([top_si, top_no])

opciones = []
for idx, row in ejemplos.iterrows():
    label = f"ID {idx} | Prob: {row['probabilidad']:.2f} | Real: {row['real']}"
    opciones.append((label, idx))
label_list = [x[0] for x in opciones]
idx_list = [x[1] for x in opciones]

selected_label = st.selectbox("Selecciona un ejemplo para explicar:", label_list)
idx_elegido = idx_list[label_list.index(selected_label)]

ejemplo = X_test.loc[[idx_elegido]]
ejemplo_scaled = scaler.transform(ejemplo)

try:
    X_sample = X_test.sample(n=1, random_state=0)
    explainer = shap.KernelExplainer(modelo.predict, X_sample)
    shap_values = explainer.shap_values(ejemplo_scaled, nsamples=100)

    shap_vals = np.array(shap_values)
    if shap_vals.ndim == 2:
        shap_vals_row = shap_vals[0]
    else:
        shap_vals_row = shap_vals

    exp = shap.Explanation(
        values=shap_vals_row.flatten(),
        base_values=float(np.array(explainer.expected_value).flatten()[0]),
        data=ejemplo_scaled.flatten(),
        feature_names=list(X_test.columns)
    )

    st.markdown(f"#### Ejemplo seleccionado: {selected_label}")
    st.dataframe(ejemplo)
    plt.gcf().set_size_inches(10, 6)
    shap.plots.bar(exp, max_display=10, show=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.warning(f"No se pudo generar el gráfico SHAP: {e}")