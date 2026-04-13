"""
E-Commerce Order Cancellation Prediction with XGBoost
======================================================
Predicts whether an incoming order will be cancelled using XGBoost
with k-fold cross-validation. The pipeline covers class balancing,
feature engineering on datetime fields, one-hot encoding of categorical
variables, and accuracy reporting.

Dataset: OnlineRetail-clean.csv
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the retail orders dataset."""
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


# ── Label Creation ────────────────────────────────────────────────────────────

def create_cancellation_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsCancellation column: True when InvoiceNo starts with 'C'."""
    df["IsCancellation"] = df["InvoiceNo"].str.startswith("C")
    print(f"\nCancellations    : {df['IsCancellation'].sum()}")
    print(f"Non-cancellations: {(~df['IsCancellation']).sum()}")
    return df


# ── Class Balancing ───────────────────────────────────────────────────────────

def balance_and_sample(df: pd.DataFrame, max_rows: int = 10_000,
                        random_state: int = 12345) -> pd.DataFrame:
    """
    Under-sample the majority class to match the minority, then reduce
    the combined dataset to fit within the max_rows memory constraint.
    """
    cancels = df[df["IsCancellation"]]
    orders  = df[~df["IsCancellation"]].sample(n=len(cancels), random_state=random_state)

    df_balanced = pd.concat([orders, cancels], axis=0)

    if len(df_balanced) > max_rows:
        df_balanced = df_balanced.sample(frac=0.5, random_state=random_state)

    print(f"\nBalanced dataset size: {len(df_balanced)}")
    return df_balanced


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-informative columns, parse InvoiceDate, and extract
    Month, Day, and Hour as numeric features.
    """
    df = df.drop(
        columns=["InvoiceNo", "Description", "Quantity", "ImputedCustomerID"],
        errors="ignore",
    )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"]   = df["InvoiceDate"].dt.day
    df["Hour"]  = df["InvoiceDate"].dt.hour
    df = df.drop(columns="InvoiceDate")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Country and StockCode columns."""
    df = pd.get_dummies(df, columns=["Country", "StockCode"])
    return df


# ── Model Training and Evaluation ─────────────────────────────────────────────

def train_and_cross_validate(X: pd.DataFrame, y: pd.Series,
                              params: dict, n_folds: int = 3,
                              num_boost_round: int = 5,
                              seed: int = 12345) -> pd.DataFrame:
    """
    Build an XGBoost DMatrix, run k-fold cross-validation,
    and return the CV results dataframe.
    """
    dmatrix = xgb.DMatrix(data=X, label=y)
    cv_results = xgb.cv(
        dtrain=dmatrix,
        params=params,
        nfold=n_folds,
        num_boost_round=num_boost_round,
        metrics="error",
        as_pandas=True,
        seed=seed,
    )
    return cv_results


def report_accuracy(cv_results: pd.DataFrame):
    """Print the cross-validated accuracy from the CV results."""
    accuracy = (1 - cv_results["test-error-mean"].iloc[-1])
    print(f"\nCross-validated Accuracy: {accuracy:.4f}")
    print("\nFull CV Results:")
    print(cv_results.to_string())
    return accuracy


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "OnlineRetail-clean.csv"

    df = load_data(DATA_PATH)
    df = create_cancellation_label(df)
    df = balance_and_sample(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    params = {"objective": "binary:logistic", "max_depth": 3}
    cv_results = train_and_cross_validate(X, y, params=params)
    report_accuracy(cv_results)
