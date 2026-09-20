"""
Predicción de la Tasa de Deserción Escolar por ETC - Proyecto Integrador CRISP-DM
Fuente: MEN - Estadísticas en Educación Preescolar, Básica y Media (Datos Abiertos Colombia)

App de despliegue en Streamlit del modelo entrenado en Google Colab
(ver notebook del proyecto para el desarrollo completo).
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de Deserción Escolar - MEN",
    page_icon="🎒",
    layout="centered",
)


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_final_ridge.pkl")


@st.cache_data
def cargar_metadata():
    with open("metadata_app.json", encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
meta = cargar_metadata()

st.title("🎒 Predicción de Tasa de Deserción Escolar")
st.caption(
    "Proyecto Integrador CRISP-DM · MEN - Estadísticas en Educación Preescolar, "
    "Básica y Media por Entidad Territorial Certificada (ETC) · Datos Abiertos Colombia"
)

st.markdown(
    """
Esta aplicación usa un modelo de **Regresión Lineal regularizada (Ridge)**, ajustado
con validación cruzada, para estimar la **tasa de deserción escolar** (%) de una
Entidad Territorial Certificada (ETC) a partir de sus indicadores de cobertura,
matriculación y repitencia.

> ⚠️ El modelo explica alrededor de un tercio de la variabilidad real de la deserción
> (R² ≈ 0.32). Es una herramienta exploratoria para priorizar ETC de riesgo relativo,
> no una predicción precisa — factores socioeconómicos no incluidos en estos datos
> también influyen fuertemente en la deserción.
"""
)

with st.form("formulario_prediccion"):
    st.subheader("Datos de la ETC")

    col1, col2 = st.columns(2)
    with col1:
        anio = st.number_input("Año", min_value=2011, max_value=2026, value=2024, step=1)
        etc = st.selectbox("Entidad Territorial Certificada (ETC)", meta["ETC"])
        poblacion = st.number_input(
            "Población en edad escolar (5-16 años)", min_value=100, max_value=2000000, value=50000, step=1000
        )
        tasa_matriculacion = st.slider("Tasa de matriculación 5-16 (%)", 0.0, 150.0, 90.0)

    with col2:
        st.markdown("**Cobertura neta (%)**")
        cob_neta = st.slider("Total", 0.0, 200.0, 80.0, key="cn")
        cob_neta_trans = st.slider("Transición", 0.0, 200.0, 65.0, key="cnt")
        cob_neta_prim = st.slider("Primaria", 0.0, 200.0, 85.0, key="cnp")
        cob_neta_sec = st.slider("Secundaria", 0.0, 200.0, 75.0, key="cns")
        cob_neta_media = st.slider("Media", 0.0, 200.0, 50.0, key="cnm")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Cobertura bruta (%)**")
        cob_bruta = st.slider("Total ", 0.0, 250.0, 95.0, key="cb")
        cob_bruta_trans = st.slider("Transición ", 0.0, 250.0, 80.0, key="cbt")
        cob_bruta_prim = st.slider("Primaria ", 0.0, 250.0, 95.0, key="cbp")
        cob_bruta_sec = st.slider("Secundaria ", 0.0, 250.0, 100.0, key="cbs")
        cob_bruta_media = st.slider("Media ", 0.0, 250.0, 85.0, key="cbm")

    with col4:
        st.markdown("**Repitencia (%)**")
        rep = st.slider("Total  ", 0.0, 40.0, 5.0, key="r")
        rep_trans = st.slider("Transición  ", 0.0, 40.0, 2.0, key="rt")
        rep_prim = st.slider("Primaria  ", 0.0, 40.0, 5.0, key="rp")
        rep_sec = st.slider("Secundaria  ", 0.0, 40.0, 8.0, key="rs")
        rep_media = st.slider("Media  ", 0.0, 40.0, 3.0, key="rm")

    enviado = st.form_submit_button("Predecir")

if enviado:
    entrada = pd.DataFrame([{
        "AÑO": anio,
        "ETC": etc,
        "POBLACIÓN_5_16": poblacion,
        "TASA_MATRICULACIÓN_5_16": tasa_matriculacion,
        "COBERTURA_NETA": cob_neta,
        "COBERTURA_NETA_TRANSICIÓN": cob_neta_trans,
        "COBERTURA_NETA_PRIMARIA": cob_neta_prim,
        "COBERTURA_NETA_SECUNDARIA": cob_neta_sec,
        "COBERTURA_NETA_MEDIA": cob_neta_media,
        "COBERTURA_BRUTA": cob_bruta,
        "COBERTURA_BRUTA_TRANSICIÓN": cob_bruta_trans,
        "COBERTURA_BRUTA_PRIMARIA": cob_bruta_prim,
        "COBERTURA_BRUTA_SECUNDARIA": cob_bruta_sec,
        "COBERTURA_BRUTA_MEDIA": cob_bruta_media,
        "REPITENCIA": rep,
        "REPITENCIA_TRANSICIÓN": rep_trans,
        "REPITENCIA_PRIMARIA": rep_prim,
        "REPITENCIA_SECUNDARIA": rep_sec,
        "REPITENCIA_MEDIA": rep_media,
    }])

    pred = modelo.predict(entrada)[0]
    pred = max(0.0, float(pred))  # la deserción no puede ser negativa

    st.subheader("Resultado")
    if pred >= 6:
        st.error(f"⚠️ Tasa de deserción predicha: **{pred:.2f}%** (riesgo alto relativo al promedio nacional)")
    elif pred >= 3.5:
        st.warning(f"Tasa de deserción predicha: **{pred:.2f}%** (riesgo moderado)")
    else:
        st.success(f"✅ Tasa de deserción predicha: **{pred:.2f}%** (riesgo bajo relativo)")

    st.caption(
        "Referencia: la tasa de deserción promedio histórica en el dataset es ≈3.75%, "
        "con un rango típico entre 0% y 14.6%."
    )

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(entrada.T.rename(columns={0: "valor"}))

st.divider()
st.caption(
    "Desarrollado como parte del Proyecto Integrador con metodología CRISP-DM. "
    "Modelo: Regresión Lineal regularizada (Ridge, alpha=1), ajustada con GridSearchCV y "
    "validación cruzada K-Fold de 5 particiones. R² en prueba: 0.321 · MAE: 0.99 puntos porcentuales."
)
