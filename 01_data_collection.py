import pandas as pd
from chembl_webresource_client.new_client import new_client

print("Target set to Human Acetylcholinesterase (CHEMBL220)")
selected_target = 'CHEMBL220'

print("Fetching bioactivity data (this might take a few minutes as the dataset is large)...")
activity = new_client.activity

# Extract data with stricter filtering for better model quality
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

# Convert to DataFrame
df = pd.DataFrame.from_dict(res)

print("Cleaning data...")
# Retain key columns
df_cleaned = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]

# Drop empty rows
df_cleaned = df_cleaned.dropna(subset=['canonical_smiles', 'standard_value'])

# Convert the standard_value column from String to Float
df_cleaned['standard_value'] = pd.to_numeric(df_cleaned['standard_value'], errors='coerce')

# Drop rows with invalid numeric values
df_cleaned = df_cleaned.dropna(subset=['standard_value'])

# Save cleaned data
df_cleaned.to_csv('bioactivity_data_raw.csv', index=False)

print("Data successfully fetched and saved as 'bioactivity_data_raw.csv'")
print(f"Total compounds ready for Machine Learning: {len(df_cleaned)}")
