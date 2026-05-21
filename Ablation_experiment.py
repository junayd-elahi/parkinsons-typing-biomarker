import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("User_Features.csv")

feature_cols = [col for col in df.columns if col not in ["user_id", "pd_label", "keystroke_count"]]
y = df["pd_label"].astype(int)
groups = df["user_id"]

central_tendency = []
variability = []

for col in ["Holdtime", "Latency", "Flighttime"]:
    central_tendency += [f"{col}_mean", f"{col}_median"]
    variability += [f"{col}_std", f"{col}_skew", f"{col}_kurtosis", f"{col}_iqr",
                    f"{col}_cv", f"{col}_mad", f"{col}_q90_q10", f"{col}_burstiness"]

central_tendency += ["Holdtime_L_mean", "Holdtime_R_mean"]
variability += ["Holdtime_L_std", "Holdtime_R_std", "hand_asymmetry"]

for direction in ["LL", "LR", "RL", "RR"]:
    central_tendency += [f"Holdtime_{direction}_mean", f"Latency_{direction}_mean",
                         f"Flighttime_{direction}_mean"]
    variability += [f"Holdtime_{direction}_std"]

central_tendency = [f for f in central_tendency if f in feature_cols]
variability = [f for f in variability if f in feature_cols]
combined = [f for f in feature_cols]

print(f"Central tendency features: {len(central_tendency)}")
print(f"Variability features: {len(variability)}")
print(f"Combined features: {len(combined)}")

overlap = set(central_tendency) & set(variability)
if overlap:
    print(f"overlapping features: {overlap}")
else:
    print("No overlap between feature groups confirmed.")

uncovered = set(feature_cols) - set(central_tendency) - set(variability)
if uncovered:
    print(f"Features not in either group: {uncovered}")



def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

conditions = {
    "A_central_tendency": central_tendency,
    "B_variability": variability,
    "C_combined": combined
}

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

inner_cv = GroupKFold(n_splits=3)
outer_cv = GroupKFold(n_splits=5)

all_results = []

for condition_name, condition_features in conditions.items():
    print(f"\nCondition: {condition_name} ({len(condition_features)} features)")

    X_cond = df[condition_features]

    rf_results = []
    xgb_results = []

    rf_cm_total = np.zeros((2, 2), dtype=int)
    xgb_cm_total = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_cond, y, groups), start=1):
        X_train = X_cond.iloc[train_idx]
        X_test = X_cond.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]

        print(
            f"Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}, "
            f"Test PD={y_test.sum()}, Test Control={(y_test == 0).sum()}"
        )

        rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
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
            "best_inner_f1": rf_grid.best_score_,
            "best_params": str(rf_grid.best_params_)
        })

        xgb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
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
            "best_inner_f1": xgb_grid.best_score_,
            "best_params": str(xgb_grid.best_params_)
        })

    rf_df = pd.DataFrame(rf_results)
    xgb_df = pd.DataFrame(xgb_results)

    print(f"\n{condition_name} - RF Average:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {metric}: {rf_df[metric].mean():.4f} (+/- {rf_df[metric].std():.4f})")

    print(
        f"RF Confusion Matrix Total: "
        f"TN={rf_cm_total[0,0]} FP={rf_cm_total[0,1]} "
        f"FN={rf_cm_total[1,0]} TP={rf_cm_total[1,1]}"
    )

    print(f"\n{condition_name} - XGBoost Average:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {metric}: {xgb_df[metric].mean():.4f} (+/- {xgb_df[metric].std():.4f})")

    print(
        f"XGB Confusion Matrix Total: "
        f"TN={xgb_cm_total[0,0]} FP={xgb_cm_total[0,1]} "
        f"FN={xgb_cm_total[1,0]} TP={xgb_cm_total[1,1]}"
    )

    for model_name, model_df, cm in [
        ("RF", rf_df, rf_cm_total),
        ("XGBoost", xgb_df, xgb_cm_total)
    ]:
        all_results.append({
            "condition": condition_name,
            "model": model_name,
            "n_features": len(condition_features),
            "accuracy": model_df["accuracy"].mean(),
            "accuracy_std": model_df["accuracy"].std(),
            "f1": model_df["f1"].mean(),
            "f1_std": model_df["f1"].std(),
            "pr_auc": model_df["pr_auc"].mean(),
            "pr_auc_std": model_df["pr_auc"].std(),
            "precision": model_df["precision"].mean(),
            "recall": model_df["recall"].mean(),
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1]
        })

summary = pd.DataFrame(all_results)

with pd.ExcelWriter("ablation_summary.xlsx", engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)

print("ablation summary finished")

print("\n ablation summary:")
print(summary.to_string(index=False))
