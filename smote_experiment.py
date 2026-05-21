import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv("User_Features.csv")

feature_cols = [col for col in df.columns if col not in ["user_id", "pd_label", "keystroke_count"]]
X = df[feature_cols]
y = df["pd_label"].astype(int)
groups = df["user_id"]

print(f"Overall: PD={y.sum()}, Control={(y==0).sum()}")
print(f"Features: {len(feature_cols)}")


def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

rf_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__class_weight": ["balanced"]
}

xgb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__scale_pos_weight": [1]
}

outer_cv = GroupKFold(n_splits=5)

rf_results = []
xgb_results = []

rf_cm_total = np.zeros((2, 2), dtype=int)
xgb_cm_total = np.zeros((2, 2), dtype=int)

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    print(
        f"\nFold {fold}: Train={len(train_idx)}, Test={len(test_idx)}, "
        f"Train PD={y_train.sum()}, Train Control={(y_train == 0).sum()}, "
        f"Test PD={y_test.sum()}, Test Control={(y_test == 0).sum()}"
    )

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42))
    ])

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        cv=GroupKFold(n_splits=3),
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    rf_grid.fit(X_train, y_train, groups=groups_train)

    rf_pred = rf_grid.best_estimator_.predict(X_test)
    rf_prob = rf_grid.best_estimator_.predict_proba(X_test)[:, 1]

    rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])
    rf_cm_total += rf_cm

    rf_results.append({
        "fold": fold,
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_test, rf_prob),
        "pr_auc": average_precision_score(y_test, rf_prob),
        "TN": int(rf_cm[0, 0]),
        "FP": int(rf_cm[0, 1]),
        "FN": int(rf_cm[1, 0]),
        "TP": int(rf_cm[1, 1]),
        "best_inner_f1": rf_grid.best_score_,
        "best_params": str(rf_grid.best_params_)
    })

    print(f"  RF best params: {rf_grid.best_params_}")
    print(f"  RF CM: TN={rf_cm[0,0]} FP={rf_cm[0,1]} FN={rf_cm[1,0]} TP={rf_cm[1,1]}")

    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])

    xgb_grid = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=GroupKFold(n_splits=3),
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    xgb_grid.fit(X_train, y_train, groups=groups_train)

    xgb_pred = xgb_grid.best_estimator_.predict(X_test)
    xgb_prob = xgb_grid.best_estimator_.predict_proba(X_test)[:, 1]

    xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    xgb_cm_total += xgb_cm

    xgb_results.append({
        "fold": fold,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred, zero_division=0),
        "recall": recall_score(y_test, xgb_pred, zero_division=0),
        "f1": f1_score(y_test, xgb_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_test, xgb_prob),
        "pr_auc": average_precision_score(y_test, xgb_prob),
        "TN": int(xgb_cm[0, 0]),
        "FP": int(xgb_cm[0, 1]),
        "FN": int(xgb_cm[1, 0]),
        "TP": int(xgb_cm[1, 1]),
        "best_inner_f1": xgb_grid.best_score_,
        "best_params": str(xgb_grid.best_params_)
    })

    print(f"  XGB best params: {xgb_grid.best_params_}")
    print(f"  XGB CM: TN={xgb_cm[0,0]} FP={xgb_cm[0,1]} FN={xgb_cm[1,0]} TP={xgb_cm[1,1]}")

rf_df = pd.DataFrame(rf_results)
xgb_df = pd.DataFrame(xgb_results)

print("\n SMOTE RF Average:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    print(f"  {metric}: {rf_df[metric].mean():.4f} (+/- {rf_df[metric].std():.4f})")

print(
    f"RF Confusion Matrix Total: "
    f"TN={rf_cm_total[0,0]} FP={rf_cm_total[0,1]} "
    f"FN={rf_cm_total[1,0]} TP={rf_cm_total[1,1]}"
)

print("\n SMOTE XGBoost Average:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    print(f"  {metric}: {xgb_df[metric].mean():.4f} (+/- {xgb_df[metric].std():.4f})")

print(
    f"XGB Confusion Matrix Total: "
    f"TN={xgb_cm_total[0,0]} FP={xgb_cm_total[0,1]} "
    f"FN={xgb_cm_total[1,0]} TP={xgb_cm_total[1,1]}"
)


summary = pd.DataFrame([
    {
        "model": "RF",
        "accuracy": rf_df["accuracy"].mean(),
        "accuracy_std": rf_df["accuracy"].std(),
        "f1": rf_df["f1"].mean(),
        "f1_std": rf_df["f1"].std(),
        "pr_auc": rf_df["pr_auc"].mean(),
        "pr_auc_std": rf_df["pr_auc"].std(),
        "precision": rf_df["precision"].mean(),
        "recall": rf_df["recall"].mean(),
        "TN": rf_cm_total[0, 0],
        "FP": rf_cm_total[0, 1],
        "FN": rf_cm_total[1, 0],
        "TP": rf_cm_total[1, 1]
    },
    {
        "model": "XGBoost",
        "accuracy": xgb_df["accuracy"].mean(),
        "accuracy_std": xgb_df["accuracy"].std(),
        "f1": xgb_df["f1"].mean(),
        "f1_std": xgb_df["f1"].std(),
        "pr_auc": xgb_df["pr_auc"].mean(),
        "pr_auc_std": xgb_df["pr_auc"].std(),
        "precision": xgb_df["precision"].mean(),
        "recall": xgb_df["recall"].mean(),
        "TN": xgb_cm_total[0, 0],
        "FP": xgb_cm_total[0, 1],
        "FN": xgb_cm_total[1, 0],
        "TP": xgb_cm_total[1, 1]
    }
])


rf_df_tagged = rf_df.assign(model="RF")
xgb_df_tagged = xgb_df.assign(model="XGBoost")
per_fold = pd.concat([rf_df_tagged, xgb_df_tagged], ignore_index=True)


with pd.ExcelWriter("smote_summary.xlsx", engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    per_fold.to_excel(writer, sheet_name="per_fold_results", index=False)

print("SMOTE summary finished")

print("\nSMOTE summary:")
print(summary.to_string(index=False))