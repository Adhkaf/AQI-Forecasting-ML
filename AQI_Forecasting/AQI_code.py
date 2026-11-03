import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

folder = r"C:\Users\VAISH\Downloads\chandigarh"

# Find original monthly files
files = glob.glob(
    os.path.join(folder, "AQI_Data*.xlsx")
)

print("Files found:", len(files))

all_data = []

for file in files:

    # Read Excel file
    df = pd.read_excel(file)

    # Extract month and year from filename
    filename = os.path.basename(file)

    match = re.search(
        r'_([A-Za-z]+)_(\d{4})_Hourly',
        filename
    )

    if not match:
        print("Skipping unexpected filename:", filename)
        continue

    month_name = match.group(1)
    year = int(match.group(2))

    month = pd.to_datetime(
        month_name,
        format="%B"
    ).month

    # Add date information
    df["Month"] = month
    df["Year"] = year

    all_data.append(df)

# Merge all monthly files
combined_df = pd.concat(
    all_data,
    ignore_index=True
)

print("Merged shape:", combined_df.shape)

# Convert wide format to long format
df_long = combined_df.melt(
    id_vars=["Date", "Month", "Year"],
    var_name="Time",
    value_name="AQI"
)

# Create timestamp
df_long["Timestamp"] = pd.to_datetime({
    "year": df_long["Year"],
    "month": df_long["Month"],
    "day": df_long["Date"]
}) + pd.to_timedelta(
    df_long["Time"].str[:2].astype(int),
    unit="h"
)

# Convert AQI into numeric format
df_long["AQI"] = pd.to_numeric(
    df_long["AQI"],
    errors="coerce"
)

# Select required columns
df_long = df_long[
    ["Timestamp", "AQI"]
]

# Sort records chronologically
df_long = df_long.sort_values(
    "Timestamp"
).reset_index(drop=True)

# Display results
print("\nFirst 10 records:")
print(df_long.head(10))

print("\nDataset shape:")
print(df_long.shape)

print("\nMissing values:")
print(df_long.isnull().sum())

# Save processed dataset
df_long.to_csv(
    os.path.join(folder, "Chandigarh_AQI_Long.csv"),
    index=False
)

print("\nTransformation completed successfully!")
# STEP 1: Check missing values
print("\n--- MISSING VALUES ---")

print(df_long.isnull().sum())

missing_data = df_long[df_long["AQI"].isnull()]

print("\nMissing AQI records:")
print(missing_data)


# STEP 2: Check duplicate timestamps
print("\n--- DUPLICATES ---")

duplicates = df_long["Timestamp"].duplicated().sum()

print("Duplicate timestamps:", duplicates)


# STEP 3: Check AQI range and statistics
print("\n--- AQI RANGE ---")

print("Minimum AQI:", df_long["AQI"].min())
print("Maximum AQI:", df_long["AQI"].max())

print("\nStatistical summary:")
print(df_long["AQI"].describe())


# STEP 4: Check missing values month-wise
print("\n--- MONTHLY MISSING VALUES ---")

df_long["Month"] = df_long["Timestamp"].dt.month

monthly_missing = df_long.groupby("Month")["AQI"].apply(
    lambda x: x.isnull().sum()
)

print(monthly_missing)
# Create AQI category

def classify_aqi(aqi):

    if pd.isna(aqi):
        return None

    elif aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


df_long["AQI_Category"] = df_long["AQI"].apply(classify_aqi)

# Check category distribution
print(df_long["AQI_Category"].value_counts())

# Calculate category percentages
print(
    df_long["AQI_Category"]
    .value_counts(normalize=True) * 100
)



plt.figure(figsize=(9, 5))

sns.countplot(
    data=df_long,
    x="AQI_Category",
    order=["Good", "Satisfactory", "Moderate",
           "Poor", "Very Poor", "Severe"]
)

plt.title("Distribution of AQI Categories")
plt.xlabel("AQI Category")
plt.ylabel("Number of Observations")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

df_long["Hour"] = df_long["Timestamp"].dt.hour

hourly_aqi = df_long.groupby("Hour")["AQI"].mean()

plt.figure(figsize=(10, 5))

plt.plot(hourly_aqi.index, hourly_aqi.values, marker="o")

plt.title("Average AQI by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")
plt.xticks(range(0, 24))

plt.grid()
plt.show()

df_long["Month"] = df_long["Timestamp"].dt.month

monthly_aqi = df_long.groupby("Month")["AQI"].mean()

plt.figure(figsize=(10, 5))

plt.plot(monthly_aqi.index, monthly_aqi.values, marker="o")

plt.title("Monthly Average AQI Trend")
plt.xlabel("Month")
plt.ylabel("Average AQI")

plt.xticks(range(1, 10))
plt.grid()
plt.show()

#Feature Engineering
# Sort data chronologically
df_long = df_long.sort_values("Timestamp").reset_index(drop=True)

# Time-based features
df_long["Hour"] = df_long["Timestamp"].dt.hour
df_long["Day"] = df_long["Timestamp"].dt.day
df_long["Month"] = df_long["Timestamp"].dt.month
df_long["DayOfWeek"] = df_long["Timestamp"].dt.dayofweek

# Previous AQI readings
df_long["AQI_Lag_1"] = df_long["AQI"].shift(1)
df_long["AQI_Lag_2"] = df_long["AQI"].shift(2)
df_long["AQI_Lag_3"] = df_long["AQI"].shift(3)
df_long["AQI_Lag_24"] = df_long["AQI"].shift(24)

# Target: category of the next hour
df_long["Target_Category"] = df_long["AQI_Category"].shift(-1)

# Check the new dataset
print(df_long.head(30))

print("\nMissing values:")
print(df_long.isnull().sum())

print(df_long.head(10))
print(df_long[["AQI", "AQI_Lag_1", "AQI_Lag_24", "Target_Category"]].tail(10))

# Sort chronologically
df_long = df_long.sort_values("Timestamp").reset_index(drop=True)

features = [
    "AQI",
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "AQI_Lag_1",
    "AQI_Lag_2",
    "AQI_Lag_3",
    "AQI_Lag_24"
]

# Keep timestamp for chronological splitting
model_df = df_long[
    ["Timestamp"] + features + ["Target_Category"]
].dropna().reset_index(drop=True)

# Calculate 80/20 split
split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index]
test_df = model_df.iloc[split_index:]

# Separate features and target
X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["Target_Category"]
y_test = test_df["Target_Category"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining category distribution:")
print(y_train.value_counts())

print("\nTesting category distribution:")
print(y_test.value_counts())

#Train_Test_Split
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    )
}

results = []

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nModel:", name)
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# Comparison table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.sort_values(
    by="F1 Score", ascending=False
))