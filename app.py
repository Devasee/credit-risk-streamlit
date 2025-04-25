import streamlit as st
import pandas as pd
import numpy as np
import shap
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.title("💳 Credit Risk Prediction App (German Credit Dataset)")

# --- Load and preprocess the data ---
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv", index_col=0)
    df.fillna("unknown", inplace=True)

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Housing'] = df['Housing'].map({'own': 0, 'rent': 1, 'free': 2})
    df['Saving accounts'] = df['Saving accounts'].map({
        'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3, 'unknown': -1
    })
    df['Checking account'] = df['Checking account'].map({
        'none': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'unknown': -1
    })

    # Cap outliers
    for col in ['Age', 'Credit amount', 'Duration']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper,
                           np.where(df[col] < lower, lower, df[col]))

    # Feature engineering
    df['Credit_per_age'] = df['Credit amount'] / df['Age']
    df['Credit_per_duration'] = df['Credit amount'] / df['Duration']

    # Target assignment
    def assign_risk(row):
        score = 0
        if row['Credit_per_duration'] > 225:
            score += 2
        if row['Credit_per_age'] > 200:
            score += 2
        if row['Credit amount'] > 5000:
            score += 2
        if row['Duration'] > 24:
            score += 2
        if row['Saving accounts'] in [0, -1]:
            score += 0
        if row['Checking account'] in [0, 1, -1]:
            score += 1
        if row['Age'] < 25 or row['Age'] > 60:
            score += 1
        if row['Housing'] == 0:
            score += 1
        if row['Job'] == 0:
            score += 1
        return 'bad' if score >= 5 else 'good'

    df['Risk'] = df.apply(assign_risk, axis=1)
    df['y'] = LabelEncoder().fit_transform(df['Risk'])

    return df

df = load_data()

# --- Train Model ---
@st.cache_resource
def train_model(df):
    X = df.drop(['Risk', 'y'], axis=1)
    X['Purpose'] = X['Purpose'].astype("category")
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scale_pos_weight = np.sum(y_train == 1) / np.sum(y_train == 0)

    params = {
        'n_estimators': [100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        random_state=42
    )

    grid = GridSearchCV(model, params, cv=StratifiedKFold(n_splits=5),
                        scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return best_model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(df)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader("📊 Model Evaluation")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
with col2:
    st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'], ax=ax)
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC curve")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# --- SHAP Interpretability ---
st.subheader("🧠 SHAP Model Interpretation")

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

st.write("### SHAP Summary Plot")
fig = plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

def get_shap_force_plot_html(force_plot):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        shap.save_html(tmpfile.name, force_plot)
        tmpfile.seek(0)
        html_content = tmpfile.read().decode()
    os.unlink(tmpfile.name)
    return html_content

sample_idx = st.number_input("Select a test sample index:", 0, len(X_test)-1, 0)
st.write("### SHAP Force Plot")
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X_test.iloc[sample_idx], matplotlib=False)
html = get_shap_force_plot_html(force_plot)
st.components.v1.html(html, height=400)


# --- Prediction Interface ---
st.subheader("📥 Make a Prediction")
input_data = X_test.iloc[0].copy()
input_fields = {}

for col in X_test.columns:
    if str(X_test[col].dtype) == "category":
        input_fields[col] = st.selectbox(f"{col}", X_train[col].cat.categories, index=0)
    elif df[col].nunique() < 10:
        input_fields[col] = st.selectbox(f"{col}", sorted(df[col].unique()), index=0)
    else:
        input_fields[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(input_data[col]))

if st.button("Predict Credit Risk"):
    input_df = pd.DataFrame([input_fields])
    input_df['Purpose'] = input_df['Purpose'].astype("category")
    prediction = model.predict(input_df)[0]
    risk = "Good" if prediction == 1 else "Bad"
    st.success(f"Predicted Credit Risk: **{risk}**")
