import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


data_path = "/Users/reezhan/Downloads/20251207-bq-results-with-mid-pkgs.jsonl"
df = pd.read_json(data_path, lines=True)

drop_cols = ["pkg_name", "is_spam"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df["is_spam"].astype(int)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_cols,
        )
    ],
    remainder="passthrough",
)

voter = VotingClassifier(
    estimators=[
        ("decision_tree", DecisionTreeClassifier(random_state=42)),
        ("log_reg", LogisticRegression(max_iter=1000)),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("svm", SVC(kernel="linear", probability=True, random_state=42)),
    ],
    voting="soft",
)

pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("scaler", StandardScaler()),
        ("voter", voter),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "models/ensemble.joblib")
