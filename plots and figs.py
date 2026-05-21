import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

mt_sheets = pd.read_excel("Model_training.xlsx", sheet_name=None, engine="openpyxl")
mt_summary = pd.read_excel("Model_training.xlsx", sheet_name="aggregate_summary", engine="openpyxl")
mt_per_fold_results = pd.read_excel("Model_training.xlsx", sheet_name="per_fold_results", engine="openpyxl")
mt_feature_importance = pd.read_excel("Model_training.xlsx", sheet_name="feature_importance", engine="openpyxl")
mt_rf_best_params = pd.read_excel("Model_training.xlsx", sheet_name="rf_best_params", engine="openpyxl")
mt_xgb_best_params = pd.read_excel("Model_training.xlsx", sheet_name="xgb_best_params", engine="openpyxl")

fs_sheets = pd.read_excel("Feature_stability.xlsx", sheet_name=None, engine="openpyxl")
fs_summary = pd.read_excel("Feature_stability.xlsx", sheet_name="retraining_summary", engine="openpyxl")
fs_frequency = pd.read_excel("Feature_stability.xlsx", sheet_name="frequency", engine="openpyxl")
fs_subset = pd.read_excel("Feature_stability.xlsx", sheet_name="per_fold_subsets", engine="openpyxl")


ablation_df = pd.read_excel("ablation_summary.xlsx", sheet_name="summary", engine="openpyxl")

baseline_row = mt_summary[mt_summary["model"] == "Majority Baseline"].iloc[0]
rf_primary_row = mt_summary[mt_summary["model"] == "Random Forest"].iloc[0]

rf_stable_row = fs_summary[fs_summary["model"] == "RF"].iloc[0]

#fig 1

conditions_order = ["A_central_tendency","B_variability", "C_combined"]
conditions_labels = ["A: Central tendency \n(20)", "B: Variability \n(31)", "C: Combined \n(51)"]

rf_f1 = [ablation_df.query("condition == @c and model == 'RF'")["f1"].iloc[0]for c in conditions_order]
xgb_f1 = [ablation_df.query("condition == @c and model == 'XGBoost'")["f1"].iloc[0]for c in conditions_order]

baseline_f1 = baseline_row["f1_mean"]

x = np.arange(len(conditions_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, rf_f1, width, label="Random Forest", color="blue", edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + width/2, xgb_f1, width, label="XGBoost", color="orange", edgecolor="black", linewidth=0.6)

ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(conditions_labels)
ax.set_ylim(0.7, 0.9)
ax.axhline(y=baseline_f1, color="grey", linestyle="--", linewidth=0.8, label=f"Majority Baseline ({baseline_f1:.3f})")
ax.legend(loc="upper right")

for bars in (bars1, bars2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("fig_ablation_f1.png", dpi=300)
plt.close()
print("Fig 1 saved")

#Fig 2
mt_feature_importance = mt_feature_importance.copy()
mt_feature_importance["combined_rank"] = mt_feature_importance[["rf_importance", "xgb_importance"]].max(axis=1)
top15 = mt_feature_importance.sort_values("combined_rank", ascending=False).head(15).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(top15))

ax.scatter(top15["rf_importance"], y_pos, color="blue", label="Random Forest", s=100, zorder=3, edgecolor="black",linewidth=0.6)
ax.scatter(top15["xgb_importance"], y_pos, color="orange", label="XGBoost", s=100, zorder=3, edgecolor="black",linewidth=0.6, marker="D")

for i in range(len(top15)):
    ax.plot([top15["rf_importance"].iloc[i], top15["xgb_importance"].iloc[i]], [y_pos[i], y_pos[i]], color="grey", linewidth=0.8, zorder=2)


feature_labels = top15["feature"].tolist()
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_labels, fontsize=10)
ax.set_xlabel("Mean importance (averaged across 5 outer folds)")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("fig_feature_importance_dotplot.png", dpi=300)
plt.close()
print("Fig 2 saved")

#fig 3

n_folds = 5
rf_stable_per_fold = []
xgb_stable_per_fold = []

for _, row in fs_subset.iterrows():
    rf_stable_per_fold.append(set(f.strip() for f in row["rf_stable_features"].split(",")))
    xgb_stable_per_fold.append(set(f.strip() for f in row["xgb_stable_features"].split(",")))


features_used = set()
for s in rf_stable_per_fold + xgb_stable_per_fold:
    features_used |= s
features_used = sorted(list(features_used))

rf_matrix = np.zeros((len(features_used), n_folds), dtype=int)
xgb_matrix = np.zeros((len(features_used), n_folds), dtype=int)

for i, feat in enumerate(features_used):
    for j in range(n_folds):
        if feat in rf_stable_per_fold[j]:
            rf_matrix[i, j] = 1
        if feat in xgb_stable_per_fold[j]:
            xgb_matrix[i, j] = 1

combined_freq = rf_matrix.sum(axis=1) + xgb_matrix.sum(axis=1)
keep = combined_freq >= 2
rf_matrix = rf_matrix[keep]
xgb_matrix = xgb_matrix[keep]
features_used = [f for f, k in zip(features_used, keep) if k]
combined_freq = combined_freq[keep]
sort_idx = np.argsort(-combined_freq)
rf_matrix = rf_matrix[sort_idx]
xgb_matrix = xgb_matrix[sort_idx]
features_sorted = [features_used[i] for i in sort_idx]

combined_matrix = np.hstack((rf_matrix, xgb_matrix))

fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(features_sorted))))

cmap = mcolors.ListedColormap(["lightgrey", "steelblue"])
sns.heatmap(combined_matrix, ax=ax, cmap=cmap, cbar=False, linewidths=0.8, linecolor="white", yticklabels=features_sorted, xticklabels=[f"F{i + 1}" for i in range(n_folds)] * 2)

ax.set_ylabel("Feature")
ax.axvline(x=n_folds, color="black", linewidth=2)
ax.text(n_folds / 2, -0.8, "Random Forest", ha="center", fontsize=11, fontweight="bold")
ax.text(n_folds + n_folds / 2, -0.8, "XGBoost", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_stability_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 3 saved")

#fig 4

labels = ["Control", "PD"]

cm_primary = np.array([
    [int(rf_primary_row["TN"]), int(rf_primary_row["FP"])],
    [int(rf_primary_row["FN"]), int(rf_primary_row["TP"])]
])

cm_stable = np.array([
    [int(rf_stable_row["TN"]), int(rf_stable_row["FP"])],
    [int(rf_stable_row["FN"]), int(rf_stable_row["TP"])]
])

mean_n_features = float(rf_stable_row["mean_n_features"])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(cm_primary, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size":16}, cbar=False, linewidths=1, linecolor="black")
axes[0].set_title("RF - 51 Features (Primary)", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_stable, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size":16}, cbar=False, linewidths=1, linecolor="black")
axes[1].set_title(f"RF - Stable Features (per-fold subsets, mean {mean_n_features:.0f})",fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("fig_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 4 saved")


#fig 5
freq_df = fs_frequency.sort_values("selected_in_n_outer_folds", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(freq_df))))
y_pos = np.arange(len(freq_df))

bars = ax.barh(y_pos, freq_df["selected_in_n_outer_folds"], color="steelblue", edgecolor="black", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(freq_df["feature"], fontsize=10)
ax.set_xlabel("Number of outer folds in which feature was selected")
ax.set_xlim(0, 5)
ax.set_xticks(range(0, 6))

for bar, freq in zip(bars, freq_df["selected_in_n_outer_folds"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f"{int(freq)}/5", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("fig_stable_subset_frequency.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 5 saved")