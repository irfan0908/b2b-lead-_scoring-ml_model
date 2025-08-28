import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# 1. Kaggle Dataset Download
# -----------------------------
def download_kaggle_dataset(dataset: str, download_path: str = "./data") -> str:
    """Download and extract Kaggle dataset, returning CSV file path."""
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api.dataset_download_files(dataset, path=download_path, unzip=True)

    files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV file found in dataset.")
    return os.path.join(download_path, files[0])


# -----------------------------
# 2. Data Preprocessing
# -----------------------------
def preprocess_data(df: pd.DataFrame, target_column: str):
    """Preprocess dataset: handle missing values, encode categoricals, scale features."""
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())   # numeric → median
    df[cat_cols] = df[cat_cols].fillna("Unknown")               # categorical → "Unknown"

    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # ensure consistent type

    # Check target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Split features/target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# -----------------------------
# 3. Model Training
# -----------------------------
def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


# -----------------------------
# 4. Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model with accuracy, classification report & confusion matrix."""
    y_pred = model.predict(X_test)

    print(f"\n🔹 {model_name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# -----------------------------
# 5. Main Pipeline
# -----------------------------
def main():
    dataset = "ashydv/leads-dataset"
    target_column = "Converted"   # ✅ Update if dataset uses different target

    print("📥 Downloading dataset...")
    csv_file = download_kaggle_dataset(dataset)
    df = pd.read_csv(csv_file)
    print("✅ Data loaded:", df.shape)

    print("⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    print("🤖 Training models...")
    models = train_models(X_train, y_train)

    print("📊 Evaluating models...")
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)


if __name__ == "__main__":
    main()
