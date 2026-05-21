import pandas as pd
from scipy.stats import median_abs_deviation

df = pd.read_csv("Tappy_Data_Cleaned.csv")
print(f"Loaded {len(df):,} rows, {df['user_id'].nunique():,} users")

keystroke_counts = df.groupby("user_id").size()
valid_users = keystroke_counts[keystroke_counts >= 2000].index
df = df[df['user_id'].isin(valid_users)]
print(f"Users with minimum 2000 Keystrokes: {df['user_id'].nunique()} users")

# df = df[
#     (df["Impact"] == "Mild") |
#     (df["Impact"] == "------") |
#     (df["Impact"].isna())
# ]
# print(f"After mild PD + controls filter: {df['user_id'].nunique()} users")

# df = df[df["Levadopa"] != "True"]
# print(f"After excluding Levodopa users: {df['user_id'].nunique()} users")

print(f"PD: {df.groupby('user_id')['pd_label'].first().sum()}")
print(f"Control: {(df.groupby('user_id')['pd_label'].first() == 0).sum()}")


def compute_features(group):
    features = {}

    for col in ["Holdtime", "Latency", "Flighttime"]:
        values = pd.to_numeric(group[col], errors="coerce").dropna()
        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_skew"] = values.skew()
        features[f"{col}_kurtosis"] = values.kurtosis()
        features[f"{col}_median"] = values.median()
        features[f"{col}_iqr"] = values.quantile(0.75) - values.quantile(0.25)


        mean_val = values.mean()
        std_val = values.std()
        if mean_val != 0:
            features[f"{col}_cv"] = std_val / mean_val
        else:
            features[f"{col}_cv"] = 0

        features[f"{col}_mad"] = median_abs_deviation(values, nan_policy="omit") if len(values) > 0 else 0

        features[f"{col}_q90_q10"] = values.quantile(0.9) - values.quantile(0.1) if len(values) > 0 else 0

        if (std_val + mean_val) != 0:
            features[f"{col}_burstiness"] = (std_val - mean_val) / (std_val + mean_val)
        else:
            features[f"{col}_burstiness"] = 0

    for hand in ["L", "R"]:
        hand_data = pd.to_numeric(group[group["Hand"] == hand]["Holdtime"], errors="coerce").dropna()
        features[f"Holdtime_{hand}_mean"] = hand_data.mean()
        features[f"Holdtime_{hand}_std"] = hand_data.std()

    l_mean = features.get("Holdtime_L_mean", 0)
    r_mean = features.get("Holdtime_R_mean", 0)
    if l_mean + r_mean > 0:
        features["hand_asymmetry"] = abs(l_mean - r_mean) / (l_mean + r_mean)
    else:
        features["hand_asymmetry"] = 0

    for direction in ["LL", "LR", "RL", "RR"]:
        dir_data = pd.to_numeric(group[group["Direction"] == direction]["Holdtime"], errors="coerce").dropna()
        features[f"Holdtime_{direction}_mean"] = dir_data.mean() if len(dir_data) > 0 else 0
        features[f"Holdtime_{direction}_std"] = dir_data.std() if len(dir_data) > 0 else 0

        dir_latency = pd.to_numeric(group[group["Direction"] == direction]["Latency"], errors="coerce").dropna()
        features[f"Latency_{direction}_mean"] = dir_latency.mean() if len(dir_latency) > 0 else 0

        dir_flight = pd.to_numeric(group[group["Direction"] == direction]["Flighttime"], errors="coerce").dropna()
        features[f"Flighttime_{direction}_mean"] = dir_flight.mean() if len(dir_flight) > 0 else 0

    features["keystroke_count"] = len(group)
    features["pd_label"] = group["pd_label"].iloc[0]

    return pd.Series(features)


user_features = df.groupby("user_id").apply(compute_features)
user_features = user_features.reset_index()

print(f"Feature matrix shape: {user_features.shape}")
print(f"Features per user: {user_features.shape[1] - 2}")

print("Sample:")
print(user_features.head())

print(f"PD: {(user_features['pd_label'] == 1).sum()}")
print(f"Control: {(user_features['pd_label'] == 0).sum()}")

user_features.to_csv("User_Features.csv", index=False)