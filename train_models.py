# src/train_models.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report

# -----------------------------
# 1) Load Data
# -----------------------------
df = pd.read_csv("data/titanic.csv")

# -----------------------------
# 2) Feature Engineering
# -----------------------------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]
target = "Survived"

X = df[features]
y = df[target]

# -----------------------------
# 3) Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# 4) Column Types
# -----------------------------
numeric_cols = ["Age", "Fare", "FamilySize"]
categorical_cols = ["Pclass", "Sex", "Embarked", "IsAlone"]

# -----------------------------
# 5) Preprocessing Pipelines
# -----------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# -----------------------------
# 6) Define Models & Hyperparameters
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingClassifier(random_state=42))
    ])
}

param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10]
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5]
    }
}

# -----------------------------
# 7) Train, Tune & Evaluate
# -----------------------------
best_model_name = None
best_model_score = 0
best_model_pipeline = None

for name, pipe in models.items():
    print(f"\n=== {name} ===")
    grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"F1 Score on Test: {f1:.3f}")
    
    if f1 > best_model_score:
        best_model_score = f1
        best_model_name = name
        best_model_pipeline = grid.best_estimator_

print(f"\n✅ Best Model: {best_model_name} with F1 = {best_model_score:.3f}")
