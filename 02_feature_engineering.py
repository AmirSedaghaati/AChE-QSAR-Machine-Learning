import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

print("Loading raw data...")
df = pd.read_csv('bioactivity_data_raw.csv')

# Function to convert IC50 to pIC50 (logarithmic scale)
# IC50 in the database is typically in nanomolar (nM) units.
def pIC50(value):
    try:
        molar = value * (10**-9) # Convert nanomolar to molar
        if molar > 0:
            return -np.log10(molar)
        else:
            return None
    except:
        return None

print("Converting IC50 to pIC50...")
df['pIC50'] = df['standard_value'].apply(pIC50)
df = df.dropna(subset=['pIC50']) # Drop rows where conversion failed

# Extract molecular fingerprints using RDKit
print("Calculating Morgan Fingerprints (this may take a minute or two)...")
fingerprints = []
valid_indices = []

for idx, smiles in enumerate(df['canonical_smiles']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Radius 2 and 2048 bits is a gold standard in QSAR modeling
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_arr = np.array(list(fp.ToBitString())).astype(int)
        fingerprints.append(fp_arr)
        valid_indices.append(idx)

# Retain only successfully converted molecules
df_valid = df.iloc[valid_indices].reset_index(drop=True)

# Convert the fingerprint list into a DataFrame with 2048 feature columns
fp_df = pd.DataFrame(fingerprints, columns=[f'fp_{i}' for i in range(2048)])

# Merge target values (pIC50) with features
df_final = pd.concat([df_valid[['molecule_chembl_id', 'pIC50']], fp_df], axis=1)

print("Saving preprocessed data...")
df_final.to_csv('bioactivity_data_preprocessed.csv', index=False)
print(f"Final dataset ready! Total molecules: {len(df_final)}, Total Features: {len(df_final.columns)}")
