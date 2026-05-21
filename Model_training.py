import pandas as pd
import numpy as np
import openpyxl

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("User_Features.csv")

print(f"pd_label value counts:\n{df['pd_label'].value_counts()}")
print(f"\nFirst 10 users and labels:")
print(df[["user_id","pd_label"]].head(10))

feature_cols = [col for col in df.columns if col not in ["user_id", "pd_label", "keystroke_count"]]
X = df[feature_cols]
y = df["pd_label"].astype(int)
groups = df["user_id"]

print(f"Overall: PD={y.sum()}, Control={(y==0).sum()}, PD%={y.mean()*100:.1f}%")
print(f"Features: {len(feature_cols)}")


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

def print_metric_summary(name, results_df, cm_total):
    print(f"\n{name}: Fold Results")
    print(results_df.to_string(index=False))
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {metric}: {results_df[metric].mean():.4f} (+/- {results_df[metric].std():.4f})")
    print("\nConfusion Matrix (total across folds):")
    print(f"Predicted Control  Predicted PD")
    print(f"  Actual Control {cm_total[0,0]:>5} {cm_total[0,1]:>5}")
    print(f"  Actual PD {cm_total[1,0]:>5}  {cm_total[1,1]:>5}")


outer_cv = GroupKFold(n_splits=5)

rf_results = []
xgb_results = []
baseline_results = []
fold_compistion = []

rf_cm_total = np.zeros((2,2), dtype=int)
xgb_cm_total = np.zeros((2,2), dtype=int)

rf_importances = np.zeros(len(feature_cols))
xgb_importances = np.zeros(len(feature_cols))

rf_best_params_by_fold = []
xgb_best_params_by_fold = []
rf_per_fold_importances = []
xgb_per_fold_importances = []


for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups),start=1):
    print(f"\nOuter Fold {fold}:")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    print(
        f"Train={len(train_idx)}, Test={len(test_idx)}, "
        f"Train PD={y_train.sum()}, Train Control={(y_train == 0).sum()}, "
        f"Test PD={y_test.sum()}, Test Control={(y_test == 0).sum()}"
    )

    fold_compistion.append({
        "fold": fold,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "train_pd": int(y_train.sum()),
        "train_control": int((y_train == 0).sum()),
        "test_pd": int(y_test.sum()),
        "test_control": int((y_test == 0).sum())
    })

    majority_label = y_train.mode()[0]
    baseline_pred = np.full(len(y_test), majority_label)
    baseline_prob = np.full(len(y_test), y_train.mean())

    baseline_metrics = evaluate_predictions(y_test, baseline_pred, baseline_prob)
    baseline_metrics["fold"] = fold
    baseline_results.append(baseline_metrics)

    print(
        f"  Majority baseline acc={baseline_metrics['accuracy']:.4f}, "
        f"F1={baseline_metrics['f1']:.4f}, PR-AUC={baseline_metrics['pr_auc']:.4f}"
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

    rf_best_model = rf_grid.best_estimator_
    rf_pred = rf_best_model.predict(X_test)
    rf_prob = rf_best_model.predict_proba(X_test)[:, 1]

    rf_metrics = evaluate_predictions(y_test, rf_pred, rf_prob)
    rf_metrics["fold"] = fold
    rf_metrics["best_inner_f1"] = rf_grid.best_score_
    rf_results.append(rf_metrics)

    rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])
    rf_cm_total += rf_cm

    rf_model = rf_best_model.named_steps["model"]
    rf_importances += rf_model.feature_importances_
    rf_per_fold_importances.append(rf_model.feature_importances_.copy())

    rf_best_params_by_fold.append({
        "fold": fold,
        "best_params": rf_grid.best_params_,
        "best_inner_f1": rf_grid.best_score_
    })

    print(f"  RF best params: {rf_grid.best_params_}")
    print(f"  RF CM: TN={rf_cm[0,0]} FP={rf_cm[0,1]} FN={rf_cm[1,0]} TP={rf_cm[1,1]}")


    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ))
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

    xgb_best_model = xgb_grid.best_estimator_
    xgb_pred = xgb_best_model.predict(X_test)
    xgb_prob = xgb_best_model.predict_proba(X_test)[:, 1]

    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)
    xgb_metrics["fold"] = fold
    xgb_metrics["best_inner_f1"] = xgb_grid.best_score_
    xgb_results.append(xgb_metrics)

    xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    xgb_cm_total += xgb_cm

    xgb_model = xgb_best_model.named_steps["model"]
    xgb_importances += xgb_model.feature_importances_
    xgb_per_fold_importances.append(xgb_model.feature_importances_.copy())

    xgb_best_params_by_fold.append({
        "fold": fold,
        "best_params": xgb_grid.best_params_,
        "best_inner_f1": xgb_grid.best_score_
    })

    print(f"  XGB best params: {xgb_grid.best_params_}")
    print(f"  XGB CM: TN={xgb_cm[0,0]} FP={xgb_cm[0,1]} FN={xgb_cm[1,0]} TP={xgb_cm[1,1]}")




baseline_df = pd.DataFrame(baseline_results)
rf_df = pd.DataFrame(rf_results)
xgb_df = pd.DataFrame(xgb_results)

print_metric_summary("Majority Baseline", baseline_df, confusion_matrix(y, np.ones(len(y)), labels=[0, 1]))
print_metric_summary("Random Forest  GroupKFold", rf_df, rf_cm_total)
print_metric_summary("XGBoost  GroupKFold", xgb_df, xgb_cm_total)

rf_importances = rf_importances / outer_cv.get_n_splits()
xgb_importances = xgb_importances / outer_cv.get_n_splits()

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "rf_importance": rf_importances,
    "xgb_importance": xgb_importances
}).sort_values("rf_importance", ascending=False)

print("\nFeature importance averaged across outer folds")
print(importance_df.to_string(index=False))

rf_params_df = pd.DataFrame(rf_best_params_by_fold)
xgb_params_df = pd.DataFrame(xgb_best_params_by_fold)

print("\nRF best parameters by outer fold:")
print(rf_params_df.to_string(index=False))

print("\nXGBoost best parameters by outer fold:")
print(xgb_params_df.to_string(index=False))

summary_rows = []

for model_name, model_df, cm in [
    ("Majority Baseline", baseline_df, None),
    ("Random Forest", rf_df, rf_cm_total),
    ("XGBoost", xgb_df, xgb_cm_total)
]:
    row = {"model": model_name}

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        row[f"{metric}_mean"] = model_df[metric].mean()
        row[f"{metric}_std"] = model_df[metric].std()

    if cm is not None:
        row["TN"] = cm[0, 0]
        row["FP"] = cm[0, 1]
        row["FN"] = cm[1, 0]
        row["TP"] = cm[1, 1]

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
composition_df = pd.DataFrame(fold_compistion)

baseline_renamed = baseline_df.rename(columns={
    c: f"baseline_{c}" for c in baseline_df.columns if c != "fold"
})

rf_renamed = rf_df.rename(columns={
    c: f"rf_{c}" for c in rf_df.columns if c != "fold"
})

xgb_renamed = xgb_df.rename(columns={
    c: f"xgb_{c}" for c in xgb_df.columns if c != "fold"
})

per_fold = composition_df.merge(baseline_renamed, on="fold")
per_fold = per_fold.merge(rf_renamed, on="fold")
per_fold = per_fold.merge(xgb_renamed, on="fold")

n_folds = outer_cv.get_n_splits()

rf_fold_cols = [f"rf_fold_{i+1}_importance" for i in range(n_folds)]
xgb_fold_cols = [f"xgb_fold_{i+1}_importance" for i in range(n_folds)]

rf_per_fold_df = pd.DataFrame(
    np.array(rf_per_fold_importances).T,
    columns=rf_fold_cols
)
rf_per_fold_df.insert(0, "feature", feature_cols)

xgb_per_fold_df = pd.DataFrame(
    np.array(xgb_per_fold_importances).T,
    columns=xgb_fold_cols
)
xgb_per_fold_df.insert(0, "feature", feature_cols)

importance_df = importance_df.merge(rf_per_fold_df, on="feature", how="left")
importance_df = importance_df.merge(xgb_per_fold_df, on="feature", how="left")

with pd.ExcelWriter("Model_training.xlsx", engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="aggregate_summary", index=False)
    per_fold.to_excel(writer, sheet_name="per_fold_results", index=False)
    importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
    rf_params_df.to_excel(writer, sheet_name="rf_best_params", index=False)
    xgb_params_df.to_excel(writer, sheet_name="xgb_best_params", index=False)

print("Model training file finished")

print("\nPrimary summary:")
print(summary_df.to_string(index=False))

print("\nPer-fold results:")
print(per_fold.to_string(index=False))

print("\nFeature importance with per-fold importances:")
print(importance_df.to_string(index=False))