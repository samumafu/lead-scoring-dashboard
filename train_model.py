from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ARTIFACT_PREFIX = "anaconda_projects_19d02227-8c9f-4e2f-b3f1-edbaa10b1251_"

DATA_PATH = Path("Leads.csv")
MODEL_PATH = Path(f"{ARTIFACT_PREFIX}lead_model.pkl")
SCALER_PATH = Path(f"{ARTIFACT_PREFIX}scaler.pkl")
ENCODERS_PATH = Path(f"{ARTIFACT_PREFIX}encoders.pkl")
COLUMNS_PATH = Path(f"{ARTIFACT_PREFIX}columns.pkl")

TARGET_COLUMN = "Converted"
ID_COLUMNS = ["Prospect ID", "Lead Number"]
NULL_LIMIT = 40


def _mode_or_default(series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else "Unknown"


def preparar_dataframe_anaconda(df_original):
    if TARGET_COLUMN not in df_original.columns:
        raise ValueError(f"No se encontro la columna objetivo '{TARGET_COLUMN}'.")

    df = df_original.copy()

    for col in ID_COLUMNS:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    null_percentage = (df.isnull().sum() / len(df)) * 100
    dropped_columns = null_percentage[null_percentage > NULL_LIMIT].index.tolist()
    df.drop(dropped_columns, axis=1, inplace=True)

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(_mode_or_default(df[col]))

    label_encoder = LabelEncoder()

    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            df[col] = label_encoder.fit_transform(df[col])

    return df, dropped_columns, label_encoder


def preparar_datos_app(df_original, scaler, feature_columns):
    df, dropped_columns, _ = preparar_dataframe_anaconda(df_original)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X = X[feature_columns]
    X_scaled = scaler.transform(X)

    return df, X_scaled, y, dropped_columns


def train_and_save():
    df_original = pd.read_csv(DATA_PATH)
    df, _, label_encoder = preparar_dataframe_anaconda(df_original)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    feature_columns = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    joblib.dump(model, "lead_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "encoders.pkl")
    joblib.dump(feature_columns, "columns.pkl")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("Archivos guardados: lead_model.pkl, scaler.pkl, encoders.pkl, columns.pkl")


if __name__ == "__main__":
    train_and_save()
