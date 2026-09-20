# Predicción de la Tasa de Deserción Escolar por ETC

Proyecto Integrador con metodología **CRISP-DM**. Aplicación de despliegue en Streamlit para un modelo de **regresión** que estima la **tasa de deserción escolar** (%) de una Entidad Territorial Certificada (ETC), a partir del dataset del Ministerio de Educación Nacional — **Estadísticas en Educación Preescolar, Básica y Media por ETC** (Datos Abiertos Colombia, datos.gov.co).

El desarrollo completo (entendimiento del negocio, entendimiento y preparación de los datos, modelamiento con 5 modelos de regresión clásicos + ensambles, ajuste de hiperparámetros y evaluación) está documentado en el notebook de Google Colab del proyecto. Este repositorio contiene **únicamente el despliegue**.

## Modelo

- **Algoritmo:** Regresión Lineal regularizada (Ridge), `alpha=1`
- **Preprocesamiento:** `StandardScaler` (numéricas) + `OneHotEncoder` (ETC, 97 categorías)
- **Ajuste:** `GridSearchCV` con validación cruzada K-Fold (5 particiones), optimizando R²
- **Desempeño en prueba (30%):** MAE 0.99 p.p. · RMSE 1.34 p.p. · **R² 0.321**

## Estructura del repositorio

```
.
├── app.py                      # Aplicación Streamlit
├── modelo_final_ridge.pkl      # Pipeline serializado (preprocesamiento + modelo)
├── metadata_app.json           # Lista de ETC para el menú desplegable de la app
├── requirements.txt            # Dependencias
└── README.md
```

## Ejecución local

```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público).
2. Entra a [streamlit.io/cloud](https://streamlit.io/cloud) e inicia sesión con tu cuenta de GitHub.
3. **New app** → selecciona el repositorio, la rama (`main`) y el archivo principal (`app.py`).
4. Click en **Deploy**.
5. Copia la URL pública generada y regístrala en el documento de entrega del proyecto, junto con un pantallazo de la app funcionando.

## Notas

- El modelo explica ≈32% de la variabilidad de la tasa de deserción (R²=0.321). Es una herramienta exploratoria para priorizar ETC de riesgo relativo, no una predicción precisa: factores socioeconómicos, de conflicto y de infraestructura no incluidos en este dataset también influyen fuertemente en la deserción escolar.
- Cualquier reentrenamiento del modelo debe regenerar `modelo_final_ridge.pkl` desde el notebook de Colab y reemplazarlo en este repositorio.
