import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline  # Note this import!
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os



def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def feature_engineering(df):
    df = df.copy()
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Amount_log'] = np.log1p(df['Amount'])
    df['V1_V2_ratio'] = df['V1'] / df['V2'].replace(0, 1e-6)
    return df.drop(['Time', 'Amount'], axis=1)

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
df = pd.read_csv(csv_path)
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature engineering transformer
feature_eng_transformer = FunctionTransformer(feature_engineering)

# Build pipeline with feature engineering, scaling, SMOTE and logistic regression
pipeline = ImbPipeline([
    ('feature_engineering', feature_eng_transformer),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=500, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
plot_cm(y_test, y_pred, "Pipeline with SMOTE + Logistic Regression")