import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

DATA_PATH = "featured_data.csv"

FEATURES = [
    "zscore_5",
    "zscore_10",
    "rsi_2",
    "ret_1",
    "ret_3",
    "tr_rel_10",
    "rel_vol_20",
]

DATE_COL = "date"
TARGET_COL = "target_1d"

PROB_THRESHOLD = 0.55
FILTER_THRESHOLD = -1  # for conditional model


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(["symbol", DATE_COL])
    return df


def split_data(df):
    train = df[df[DATE_COL] < "2023-01-01"].copy()
    test  = df[df[DATE_COL] >= "2023-01-01"].copy()
    return train, test


def run_experiment(df, conditional=False):

    train, test = split_data(df)

    # Optional conditional training filter
    if conditional:
        train = train[train["zscore_5"] <= FILTER_THRESHOLD]

    # Drop rows with missing target
    train = train.dropna(subset=[TARGET_COL])
    test = test.dropna(subset=[TARGET_COL])

    X_train = train[FEATURES]
    y_train = train[TARGET_COL]

    X_test = test[FEATURES]
    y_test = test[TARGET_COL]

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    test = test.copy()
    test["prob"] = probs

    # Long-only trade rule
    trades = test[
        (test["zscore_5"] <= FILTER_THRESHOLD) &
        (test["prob"] >= PROB_THRESHOLD)
    ]

    strategy_returns = trades["fwd_ret_1"]

    if len(strategy_returns) > 0:
        sharpe = (
            strategy_returns.mean() /
            strategy_returns.std()
        ) * np.sqrt(252)
    else:
        sharpe = np.nan

    print("\n==============================")
    print("Conditional Training:", conditional)
    print("AUC:", round(auc, 4))
    print("Trades:", len(trades))
    print("Avg fwd_ret_1:", round(strategy_returns.mean(), 5))
    print("Win rate:", round((strategy_returns > 0).mean(), 4))
    print("Sharpe:", round(sharpe, 3))
    print("==============================\n")


def main():
    df = load_data()

    print("Running Global Model...")
    run_experiment(df, conditional=False)

    print("Running Conditional Model...")
    run_experiment(df, conditional=True)


if __name__ == "__main__":
    main()