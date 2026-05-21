import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from train_model import (
    COLUMNS_PATH,
    ENCODERS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    preparar_datos_app
)


st.set_page_config(
    page_title="Lead Scoring Dashboard",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0f2747 50%, #09111f 100%);
    color: #f9fafb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, h5, h6, p, label, span {
    color: #f9fafb !important;
}

.main-title {
    font-size: 40px;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #d1d5db;
    margin-bottom: 25px;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.13);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.25);
}

.kpi-title {
    font-size: 14px;
    color: #cbd5e1;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 31px;
    font-weight: 900;
    color: #ffffff;
}

.kpi-note {
    font-size: 12px;
    color: #94a3b8;
}

.info-box {
    background: rgba(255, 255, 255, 0.07);
    border-left: 4px solid #38bdf8;
    padding: 15px 18px;
    border-radius: 12px;
    color: #e5e7eb;
}

div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}

.stSelectbox, .stSlider {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    return pd.read_csv("Leads.csv")


@st.cache_resource
def cargar_artifactos():
    paths = [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, COLUMNS_PATH]
    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        st.error(
            "Faltan archivos del modelo. Ejecuta primero: python train_model.py"
        )
        st.stop()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return (
            joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(ENCODERS_PATH),
            joblib.load(COLUMNS_PATH)
        )


def nombre_modelo(model):
    nombres = {
        "RandomForestClassifier": "Random Forest",
        "LogisticRegression": "Regresión Logística"
    }
    return nombres.get(type(model).__name__, type(model).__name__)


def asignar_prioridad(score):
    if score >= 0.80:
        return "Alta"
    if score >= 0.50:
        return "Media"
    return "Baja"


df_original = cargar_datos()
model, scaler, _label_encoder, feature_columns = cargar_artifactos()

try:
    df_modelo, X_scaled, y, columnas_eliminadas = preparar_datos_app(
        df_original,
        scaler,
        feature_columns
    )
except Exception as exc:
    st.error(f"No se pudieron preparar los datos para el modelo: {exc}")
    st.stop()

_, X_test, _, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
lead_scores = model.predict_proba(X_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
mejor_modelo_nombre = nombre_modelo(model)

resultados_df = pd.DataFrame([{
    "Modelo": mejor_modelo_nombre,
    "Accuracy": accuracy,
    "ROC-AUC": auc
}])

ranking = df_modelo.copy()
ranking["Lead_Score"] = lead_scores
ranking["Probabilidad Conversión (%)"] = (lead_scores * 100).round(2)
ranking["Prioridad"] = ranking["Lead_Score"].apply(asignar_prioridad)

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Variable": feature_columns,
        "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)
else:
    importance = pd.DataFrame(columns=["Variable", "Importancia"])

total_leads = len(ranking)
alta = ranking[ranking["Prioridad"] == "Alta"].shape[0]
media = ranking[ranking["Prioridad"] == "Media"].shape[0]
baja = ranking[ranking["Prioridad"] == "Baja"].shape[0]


st.markdown("""
<div class="main-title">Lead Scoring Dashboard</div>
<div class="subtitle">
Sistema predictivo para priorizar leads comerciales según su probabilidad de conversión.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-box">
Este dashboard permite al equipo comercial identificar los leads con mayor probabilidad de conversión,
clasificarlos por prioridad y ordenar el seguimiento de ventas de forma estratégica.
</div>
""", unsafe_allow_html=True)

st.write("")


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total de leads</div>
        <div class="kpi-value">{total_leads:,}</div>
        <div class="kpi-note">Registros analizados</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Prioridad Alta</div>
        <div class="kpi-value">{alta:,}</div>
        <div class="kpi-note">Contactar primero</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Prioridad Media</div>
        <div class="kpi-value">{media:,}</div>
        <div class="kpi-note">Seguimiento comercial</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">ROC-AUC</div>
        <div class="kpi-value">{auc:.3f}</div>
        <div class="kpi-note">Capacidad predictiva</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Mejor modelo</div>
        <div class="kpi-value" style="font-size:22px;">{mejor_modelo_nombre}</div>
        <div class="kpi-note">Seleccionado por ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.divider()


c1, c2, c3 = st.columns([1.1, 1.1, 1])

with c1:
    st.subheader("Distribución de prioridades")

    prioridad_counts = ranking["Prioridad"].value_counts().reindex(
        ["Alta", "Media", "Baja"]
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("#0f2747")
    ax.set_facecolor("#0f2747")

    ax.bar(
        prioridad_counts.index,
        prioridad_counts.values,
        color=["#22c55e", "#facc15", "#ef4444"]
    )

    ax.set_title("Leads por prioridad", color="white")
    ax.set_xlabel("Prioridad", color="white")
    ax.set_ylabel("Cantidad", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, width="stretch")

with c2:
    st.subheader("Comparación de modelos")

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("#0f2747")
    ax.set_facecolor("#0f2747")

    ax.bar(
        resultados_df["Modelo"],
        resultados_df["ROC-AUC"],
        color="#38bdf8"
    )

    ax.set_ylim(0, 1)
    ax.set_title("ROC-AUC por modelo", color="white")
    ax.set_ylabel("ROC-AUC", color="white")
    ax.tick_params(colors="white", labelrotation=15)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, width="stretch")

with c3:
    st.subheader("Resumen técnico")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Accuracy</div>
        <div class="kpi-value">{accuracy:.3f}</div>
        <div class="kpi-note">Porcentaje de predicciones correctas</div>
        <br>
        <div class="kpi-title">Columnas eliminadas por nulos</div>
        <div class="kpi-value" style="font-size:22px;">{len(columnas_eliminadas)}</div>
        <div class="kpi-note">Feature engineering aplicado</div>
    </div>
    """, unsafe_allow_html=True)


st.divider()


st.subheader("Ranking comercial de leads")

f1, f2 = st.columns([1, 1])

with f1:
    filtro_prioridad = st.selectbox(
        "Filtrar por prioridad",
        ["Todas", "Alta", "Media", "Baja"]
    )

with f2:
    cantidad_mostrar = st.slider(
        "Cantidad de leads a mostrar",
        min_value=10,
        max_value=300,
        value=50,
        step=10
    )

ranking_filtrado = ranking.copy()

if filtro_prioridad != "Todas":
    ranking_filtrado = ranking_filtrado[
        ranking_filtrado["Prioridad"] == filtro_prioridad
    ]

ranking_filtrado = ranking_filtrado.sort_values(
    by="Lead_Score",
    ascending=False
)
ranking_filtrado = ranking_filtrado.reset_index(drop=True)
ranking_filtrado["Lead"] = ranking_filtrado.index + 1

st.caption(
    "Cuando seleccionas 'Todas', la tabla se ordena por probabilidad de conversión. "
    "Por eso los primeros registros suelen ser de prioridad Alta."
)

columnas_tabla = []

for col in ["Lead Number", "Prospect ID", "Lead"]:
    if col in ranking_filtrado.columns:
        columnas_tabla.append(col)

columnas_tabla += [
    "Probabilidad Conversión (%)",
    "Prioridad"
]

st.dataframe(
    ranking_filtrado[columnas_tabla].head(cantidad_mostrar),
    width="stretch",
    hide_index=True
)

csv = ranking_filtrado.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar ranking en CSV",
    data=csv,
    file_name="lead_scoring_results.csv",
    mime="text/csv"
)


st.divider()


e1, e2 = st.columns(2)

with e1:
    st.subheader("Curva ROC")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("#0f2747")
    ax.set_facecolor("#0f2747")

    ax.plot(fpr, tpr, color="#38bdf8", label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")

    ax.set_title("Evaluación ROC", color="white")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0f2747", labelcolor="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, width="stretch")

with e2:
    st.subheader("Matriz de confusión")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_facecolor("#0f2747")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False
    )

    ax.set_title("Matriz de confusión", color="white")
    ax.set_xlabel("Predicción", color="white")
    ax.set_ylabel("Valor real", color="white")
    ax.tick_params(colors="white")

    st.pyplot(fig, width="stretch")


if not importance.empty:
    st.divider()
    st.subheader("Variables más importantes del modelo")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f2747")
    ax.set_facecolor("#0f2747")

    sns.barplot(
        data=importance.head(10),
        x="Importancia",
        y="Variable",
        ax=ax,
        color="#38bdf8"
    )

    ax.set_title("Factores que más influyen en la conversión", color="white")
    ax.set_xlabel("Importancia", color="white")
    ax.set_ylabel("")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, width="stretch")


st.divider()

with st.expander("Ver comparación completa de modelos"):
    st.dataframe(
        resultados_df.round(4),
        width="stretch",
        hide_index=True
    )

with st.expander("Ver muestra del dataset original"):
    st.dataframe(
        df_original.head(20),
        width="stretch",
        hide_index=True
    )

st.caption(
    "Dashboard comercial desarrollado en Streamlit para el proyecto Lead Scoring Predictivo."
)
