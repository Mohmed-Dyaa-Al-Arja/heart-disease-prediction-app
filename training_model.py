import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from itertools import combinations
import plotly.express as px

from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

data_path = "heart.csv"  
df = pd.read_csv(data_path)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

logistic = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb = XGBClassifier(scale_pos_weight=(neg/pos), use_label_encoder=False, eval_metric='logloss', random_state=42)
easy = EasyEnsembleClassifier(random_state=42, n_estimators=10)


ensemble = VotingClassifier(
    estimators=[
        ('easy', easy),
        ('logreg', logistic),
        ('xgb', xgb),
        ('rf', rf)
    ],
    voting='soft',
    weights=[2.0, 1.5, 1.0, 1.0]
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ensemble)
])


model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.45
y_pred = (y_proba >= threshold).astype(int)

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print(f"\n🎯 ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

joblib.dump(model, "best_model.pkl")
print("✅ Model saved as 'best_model.pkl'")