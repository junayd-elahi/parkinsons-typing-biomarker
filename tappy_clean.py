import os
import pandas as pd

tappy_dataset = os.path.join("Tappy Data")
archived_users = os.path.join("Archived users")
output_csv = "Tappy_Data_Cleaned.csv"

def load_user_metadata(folder_path):
    metadata_rows = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        user_data = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if ":" in line:
                    key, value = line.split(":", 1)
                    user_data[key.strip()] = value.strip()

        user_id = filename.replace ("User_","").replace(".txt","")

        pd_value = user_data.get("Parkinsons","")
        if pd_value == "True":
            pd_label = 1
        elif pd_value == "False":
            pd_label = 0
        else:
            pd_label = None

        metadata_rows.append({
            "user_id": user_id,
            "pd_label": pd_label,
            "gender": user_data.get("Gender"),
            "tremors": user_data.get("Tremors"),
            "diagnosis_year": user_data.get("DiagnosisYear"),
            "updrs": user_data.get("UPDRS"),
            "Impact": user_data.get("Impact"),
            "Levadopa": user_data.get("Levadopa"),
            "DA": user_data.get("DA"),
            "MOAB": user_data.get("MOAB"),
            "Other": user_data.get("Other")
        })

    metadata_df = pd.DataFrame(metadata_rows)
    return metadata_df

#metadata_df = load_user_metadata(archived_users)

# print(metadata_df.head())
# print(metadata_df.shape)
# print(metadata_df["pd_label"].value_counts(dropna=False))

def load_keystroke_data(folder_path):
    keystroke = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        df = pd.read_csv(
            file_path,
            sep=r"\t",
            engine="python",
            header=None,
            names=["user_id", "Date", "Time", "Hand", "Holdtime", "Direction", "Latency", "Flighttime"]
        )
        df = df.iloc[:, :8]


        df["source_file"] = filename
        keystroke.append(df)

    keystrokes_df = pd.concat(keystroke, ignore_index=True)
    return keystrokes_df

# keystrokes_df = load_keystroke_data(tappy_dataset)
# print(keystrokes_df.head())
# print(keystrokes_df.shape)
# print(keystrokes_df.columns)

def clean_data(keystrokes_df):
    raw = len(keystrokes_df)
    for col in ["Holdtime","Latency","Flighttime"]:
        keystrokes_df[col] = pd.to_numeric(keystrokes_df[col],errors="coerce")

    keystrokes_df = keystrokes_df.dropna(subset=["Holdtime","Latency","Flighttime"])
    print(f"Dropped {raw - len(keystrokes_df)} rows due to missing values")

    keystrokes_df = keystrokes_df[
        (keystrokes_df["Holdtime"] > 0) & (keystrokes_df["Holdtime"] <= 5000) &
        (keystrokes_df["Latency"] > 0) & (keystrokes_df["Latency"] <= 10000) &
        (keystrokes_df["Flighttime"] > -5000 ) & (keystrokes_df["Flighttime"] <= 10000)
    ]

    print(f"After cleaning, {len(keystrokes_df)} rows remain")
    return keystrokes_df


metadata_df = load_user_metadata(archived_users)
print(f"Loaded {len(metadata_df)} users")
print(f"PD: {(metadata_df['pd_label'] == 1).sum()}")
print(f"Control: {(metadata_df['pd_label'] == 0).sum()}")

keystroke_df = load_keystroke_data(tappy_dataset)
print(f"Loaded {len(keystroke_df)} keystrokes rows")

print ("sample of raw data:")
print(keystroke_df.head(10))

keystrokes_df = clean_data(keystroke_df)

merged = keystrokes_df.merge(
    metadata_df [["user_id","pd_label","gender","Impact","Levadopa"]],
    on="user_id",
    how="inner"
)

print(f"Merged dataset: {len(merged):,} rows")
print(f"Users matched: {merged['user_id'].nunique()}")
print(f"PD: {(merged.groupby('user_id')['pd_label'].first() == 1).sum()}")
print(f"Control: {(merged.groupby('user_id')['pd_label'].first() == 0).sum()}")

merged.to_csv(output_csv, index=False)
