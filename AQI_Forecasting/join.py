import pandas as pd
import glob
import os

folder_path = r"C:\Users\VAISH\Downloads\chandigarh"

# Read only original monthly files
files = glob.glob(
    os.path.join(
        folder_path,
        "AQI_Data_City_Level_Chandigarh_*.xlsx"
    )
)

print("Files found:", len(files))

all_data = []

for file in files:

    df = pd.read_excel(file)

    # Keep track of original file
    df["Source_File"] = os.path.basename(file)

    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

print("Combined shape:", combined_df.shape)

# Save outside the raw data folder
output_path = r"C:\Users\VAISH\Downloads\Chandigarh_Combined_Raw.xlsx"

combined_df.to_excel(output_path, index=False)

print("Merging completed!")