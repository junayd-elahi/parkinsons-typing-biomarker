import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("User_Features.csv")

y = df["pd_label"].astype(int)
groups = df["user_id"]

central_tendency = []

for col in ["Holdtime", "Latency", "Flighttime"]:
    central_tendency += [f"{col}_mean", f"{col}_median"]

central_tendency += ["Holdtime_L_mean", "Holdtime_R_mean"]

for direction in ["LL", "LR", "RL", "RR"]:
    central_tendency += [
        f"Holdtime_{direction}_mean",
        f"Latency_{direction}_mean",
        f"Flighttime_{direction}_mean"
    ]

feature_cols = [f for f in central_tendency if f in df.columns]
X = df[feature_cols]

print(f"Starting features: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print(f"Overall: PD={y.sum()}, Control={(y == 0).sum()}")


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

def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }


def make_pipeline(model_name):
    if model_name == "RF":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42))
        ])

    if model_name == "XGBoost":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
        ])

    raise ValueError("model_name must be 'RF' or 'XGBoost'")


def get_param_grid(model_name):
    if model_name == "RF":
        return rf_param_grid

    if model_name == "XGBoost":
        return xgb_param_grid

    raise ValueError("model_name must be 'RF' or 'XGBoost'")


def tune_model(X_train, y_train, groups_train, model_name):
    pipeline = make_pipeline(model_name)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(model_name),
        cv=GroupKFold(n_splits=3),
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train, groups=groups_train)
    return grid


def extract_importance(best_estimator, features):
    model = best_estimator.named_steps["model"]
    return pd.Series(model.feature_importances_, index=features)


def select_stable_features_within_training_data(
    X_train,
    y_train,
    groups_train,
    model_name,
    top_k=10,
    required_frequency=3
):


    stability_cv = GroupKFold(n_splits=3)
    top_features_per_inner_fold = []

    for inner_fold, (inner_train_idx, inner_test_idx) in enumerate(
        stability_cv.split(X_train, y_train, groups_train),
        start=1
    ):
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        groups_inner_train = groups_train.iloc[inner_train_idx]

        grid = tune_model(
            X_inner_train,
            y_inner_train,
            groups_inner_train,
            model_name
        )

        importances = extract_importance(grid.best_estimator_, X_train.columns)
        top_features = set(importances.nlargest(top_k).index)

        top_features_per_inner_fold.append(top_features)

        print(
            f"    {model_name} inner stability fold {inner_fold} "
            f"top-{top_k}: {sorted(top_features)}"
        )

    feature_frequency = {}

    for feature in X_train.columns:
        feature_frequency[feature] = sum(
            1 for fold_set in top_features_per_inner_fold
            if feature in fold_set
        )

    stable_features = sorted([
        feature for feature, frequency in feature_frequency.items()
        if frequency >= required_frequency
    ])

    if len(stable_features) == 0:
        print(
            f"    No {model_name} features appeared in all 3 inner stability folds. "
            f"Relaxing threshold to 2/3."
        )

        stable_features = sorted([
            feature for feature, frequency in feature_frequency.items()
            if frequency >= 2
        ])

    return stable_features, feature_frequency




outer_cv = GroupKFold(n_splits=5)

top_k = 10

all_results = []
selected_feature_records = []

rf_cm_total = np.zeros((2, 2), dtype=int)
xgb_cm_total = np.zeros((2, 2), dtype=int)

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    print(f"\nOuter Fold {fold}")

    X_train_full = X.iloc[train_idx]
    X_test_full = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    print(
        f"Train={len(train_idx)}, Test={len(test_idx)}, "
        f"Train PD={y_train.sum()}, Train Control={(y_train == 0).sum()}, "
        f"Test PD={y_test.sum()}, Test Control={(y_test == 0).sum()}"
    )


    rf_stable, rf_freq = select_stable_features_within_training_data(
        X_train_full,
        y_train,
        groups_train,
        model_name="RF",
        top_k=top_k,
        required_frequency=3
    )

    xgb_stable, xgb_freq = select_stable_features_within_training_data(
        X_train_full,
        y_train,
        groups_train,
        model_name="XGBoost",
        top_k=top_k,
        required_frequency=3
    )

    stable_features = sorted(set(rf_stable) | set(xgb_stable))

    if len(stable_features) == 0:
        print("  No stable features selected for this outer fold. Skipping fold.")
        continue

    print(f"  RF stable features: {rf_stable}")
    print(f"  XGBoost stable features: {xgb_stable}")
    print(f"  Union stable features: {stable_features}")

    selected_feature_records.append({
        "fold": fold,
        "rf_stable_features": ", ".join(rf_stable),
        "xgb_stable_features": ", ".join(xgb_stable),
        "union_stable_features": ", ".join(stable_features),
        "n_union_stable_features": len(stable_features)
    })


    X_train_stable = X_train_full[stable_features]
    X_test_stable = X_test_full[stable_features]

    rf_grid = tune_model(
        X_train_stable,
        y_train,
        groups_train,
        model_name="RF"
    )

    rf_pred = rf_grid.best_estimator_.predict(X_test_stable)
    rf_prob = rf_grid.best_estimator_.predict_proba(X_test_stable)[:, 1]

    rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])
    rf_cm_total += rf_cm

    rf_metrics = evaluate_predictions(y_test, rf_pred, rf_prob)
    rf_metrics.update({
        "fold": fold,
        "model": "RF",
        "n_features": len(stable_features),
        "selected_features": ", ".join(stable_features),
        "best_inner_f1": rf_grid.best_score_,
        "best_params": str(rf_grid.best_params_),
        "TN": rf_cm[0, 0],
        "FP": rf_cm[0, 1],
        "FN": rf_cm[1, 0],
        "TP": rf_cm[1, 1]
    })

    all_results.append(rf_metrics)

    print(f"  RF best params: {rf_grid.best_params_}")
    print(f"  RF CM: TN={rf_cm[0,0]} FP={rf_cm[0,1]} FN={rf_cm[1,0]} TP={rf_cm[1,1]}")



    xgb_grid = tune_model(
        X_train_stable,
        y_train,
        groups_train,
        model_name="XGBoost"
    )

    xgb_pred = xgb_grid.best_estimator_.predict(X_test_stable)
    xgb_prob = xgb_grid.best_estimator_.predict_proba(X_test_stable)[:, 1]

    xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    xgb_cm_total += xgb_cm

    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)
    xgb_metrics.update({
        "fold": fold,
        "model": "XGBoost",
        "n_features": len(stable_features),
        "selected_features": ", ".join(stable_features),
        "best_inner_f1": xgb_grid.best_score_,
        "best_params": str(xgb_grid.best_params_),
        "TN": xgb_cm[0, 0],
        "FP": xgb_cm[0, 1],
        "FN": xgb_cm[1, 0],
        "TP": xgb_cm[1, 1]
    })

    all_results.append(xgb_metrics)

    print(f"  XGBoost best params: {xgb_grid.best_params_}")
    print(f"  XGB CM: TN={xgb_cm[0,0]} FP={xgb_cm[0,1]} FN={xgb_cm[1,0]} TP={xgb_cm[1,1]}")


results_df = pd.DataFrame(all_results)
selected_features_df = pd.DataFrame(selected_feature_records)

print("\nStable feature selection by outer fold:")
print(selected_features_df.to_string(index=False))

print("\nStable feature retraining fold results:")
print(results_df.to_string(index=False))

summary_rows = []

for model_name, cm_total in [
    ("RF", rf_cm_total),
    ("XGBoost", xgb_cm_total)
]:
    model_df = results_df[results_df["model"] == model_name]

    row = {
        "model": model_name,
        "mean_n_features": model_df["n_features"].mean(),
        "accuracy": model_df["accuracy"].mean(),
        "accuracy_std": model_df["accuracy"].std(),
        "precision": model_df["precision"].mean(),
        "precision_std": model_df["precision"].std(),
        "recall": model_df["recall"].mean(),
        "recall_std": model_df["recall"].std(),
        "f1": model_df["f1"].mean(),
        "f1_std": model_df["f1"].std(),
        "roc_auc": model_df["roc_auc"].mean(),
        "roc_auc_std": model_df["roc_auc"].std(),
        "pr_auc": model_df["pr_auc"].mean(),
        "pr_auc_std": model_df["pr_auc"].std(),
        "TN": cm_total[0, 0],
        "FP": cm_total[0, 1],
        "FN": cm_total[1, 0],
        "TP": cm_total[1, 1]
    }

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print("\nFeature stability retraining summary:")
print(summary_df.to_string(index=False))

feature_frequency = {}

for feature_list in selected_features_df["union_stable_features"]:
    if pd.isna(feature_list) or feature_list == "":
        continue

    features = [f.strip() for f in feature_list.split(",")]

    for feature in features:
        feature_frequency[feature] = feature_frequency.get(feature, 0) + 1

feature_frequency_df = pd.DataFrame([
    {"feature": feature, "selected_in_n_outer_folds": count}
    for feature, count in feature_frequency.items()
]).sort_values("selected_in_n_outer_folds", ascending=False)


with pd.ExcelWriter("Feature_stability.xlsx", engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="retraining_summary", index=False)
    selected_features_df.to_excel(writer, sheet_name="per_fold_subsets", index=False)
    feature_frequency_df.to_excel(writer, sheet_name="frequency", index=False)
    results_df.to_excel(writer, sheet_name="per_fold_metrics", index=False)

print("Feature stability file finished")