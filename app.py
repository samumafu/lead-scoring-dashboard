import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve


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


@st.cache_data
def entrenar_modelos(df_original):
    df = df_original.copy()

    if "Converted" not in df.columns:
        st.error("No se encontró la columna objetivo 'Converted' en el dataset.")
        st.stop()

    ids = pd.DataFrame(index=df.index)

    if "Lead Number" in df.columns:
        ids["Lead Number"] = df["Lead Number"]

    if "Prospect ID" in df.columns:
        ids["Prospect ID"] = df["Prospect ID"]

    columnas_no_modelo = ["Prospect ID", "Lead Number"]
    df_modelo = df.drop(columns=[c for c in columnas_no_modelo if c in df.columns])

    porcentaje_nulos = (df_modelo.isnull().sum() / len(df_modelo)) * 100
    columnas_muchos_nulos = porcentaje_nulos[porcentaje_nulos > 40].index.tolist()

    if "Converted" in columnas_muchos_nulos:
        columnas_muchos_nulos.remove("Converted")

    df_modelo = df_modelo.drop(columns=columnas_muchos_nulos)

    X = df_modelo.drop("Converted", axis=1)
    y = df_modelo["Converted"]

    columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns
    columnas_categoricas = X.select_dtypes(include=["object"]).columns

    procesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                columnas_numericas
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                columnas_categoricas
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            class_weight="balanced"
        )
    }

    resultados = []

    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        pipeline = Pipeline(steps=[
            ("procesador", procesador),
            ("modelo", modelo)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

        modelos_entrenados[nombre] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

    resultados_df = pd.DataFrame(resultados).sort_values(
        by="ROC-AUC",
        ascending=False
    )

    mejor_modelo_nombre = resultados_df.iloc[0]["Modelo"]
    mejor_modelo = modelos_entrenados[mejor_modelo_nombre]["pipeline"]
    y_pred_best = modelos_entrenados[mejor_modelo_nombre]["y_pred"]
    y_prob_best = modelos_entrenados[mejor_modelo_nombre]["y_prob"]

    lead_scores = mejor_modelo.predict_proba(X)[:, 1]

    ranking = ids.copy()

    if ranking.empty:
        ranking["Lead"] = range(1, len(df_original) + 1)

    ranking["Probabilidad Conversión (%)"] = (lead_scores * 100).round(2)
    ranking["Lead_Score"] = lead_scores

    def asignar_prioridad(score):
        if score >= 0.80:
            return "Alta"
        elif score >= 0.50:
            return "Media"
        else:
            return "Baja"

    ranking["Prioridad"] = ranking["Lead_Score"].apply(asignar_prioridad)

    if mejor_modelo_nombre == "Random Forest":
        try:
            feature_names = mejor_modelo.named_steps["procesador"].get_feature_names_out()
            importancias = mejor_modelo.named_steps["modelo"].feature_importances_

            importance = pd.DataFrame({
                "Variable": feature_names,
                "Importancia": importancias
            }).sort_values(by="Importancia", ascending=False)

            importance["Variable"] = (
                importance["Variable"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
        except Exception:
            importance = pd.DataFrame(columns=["Variable", "Importancia"])
    else:
        importance = pd.DataFrame(columns=["Variable", "Importancia"])

    return (
        ranking,
        resultados_df,
        mejor_modelo_nombre,
        y_test,
        y_pred_best,
        y_prob_best,
        importance,
        columnas_muchos_nulos
    )


df_original = cargar_datos()

(
    ranking,
    resultados_df,
    mejor_modelo_nombre,
    y_test,
    y_pred,
    y_prob,
    importance,
    columnas_eliminadas
) = entrenar_modelos(df_original)


accuracy = resultados_df.iloc[0]["Accuracy"]
auc = resultados_df.iloc[0]["ROC-AUC"]

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

    st.pyplot(fig, use_container_width=True)

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

    st.pyplot(fig, use_container_width=True)

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
    use_container_width=True,
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

    st.pyplot(fig, use_container_width=True)

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

    st.pyplot(fig, use_container_width=True)


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

    st.pyplot(fig, use_container_width=True)


st.divider()

with st.expander("Ver comparación completa de modelos"):
    st.dataframe(
        resultados_df.round(4),
        use_container_width=True,
        hide_index=True
    )

with st.expander("Ver muestra del dataset original"):
    st.dataframe(
        df_original.head(20),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Dashboard comercial desarrollado en Streamlit para el proyecto Lead Scoring Predictivo."
)