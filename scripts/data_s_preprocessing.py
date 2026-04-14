#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "data/raw"
RESULTS_DIR = "results"

def parse_dc50_dmax(value):
    if pd.isna(value) or value == '':
        return np.nan
    value_str = str(value).strip()
    if value_str == '>3 uM' or value_str == '> 3 uM':
        return 3000
    if value_str == '> 90 %':
        return 90
    try:
        return float(value_str)
    except:
        return np.nan

def preprocess_protacdb(protacdb_csv):
    print("Preprocessing PROTACdb data...")
    df = protacdb_csv.copy()
    df['DC50_nM'] = df['Dc50'].apply(parse_dc50_dmax)
    df['Dmax_percent'] = df['Dmax'].apply(parse_dc50_dmax)
    df['Activity_Label'] = (df['Active/Inactive'] == 'Active').astype(int)
    df['Target_UniProt'] = df['Target']
    df['E3_Type'] = df['E3 Ligase'].fillna('Unknown')
    if 'MW' in df.columns:
        df['MW_numeric'] = pd.to_numeric(df['MW'], errors='coerce')
    
    print(f"Total compounds: {len(df)}")
    print(f"Active compounds: {int(df['Activity_Label'].sum())}")
    print(f"Inactive compounds: {int((1-df['Activity_Label']).sum())}")
    print(f"DC50 values available: {int(df['DC50_nM'].notna().sum())}")
    print(f"Dmax values available: {int(df['Dmax_percent'].notna().sum())}")
    
    return df

def integrate_datasets(protac_csv, protacdb_csv):
    print("\nIntegrating datasets...")
    protacdb_processed = preprocess_protacdb(protacdb_csv)
    
    integrated_data = {
        'total_protacs': int(len(protac_csv)),
        'labeled_protacs': int(len(protacdb_processed)),
        'active_protacs': int(protacdb_processed['Activity_Label'].sum()),
        'inactive_protacs': int((1 - protacdb_processed['Activity_Label']).sum()),
        'targets': int(protacdb_processed['Target'].nunique()),
        'e3_ligases': int(protacdb_processed['E3 Ligase'].nunique()),
    }
    
    print(f"\nIntegration Summary:")
    for key, value in integrated_data.items():
        print(f"  {key}: {value}")
    
    return protacdb_processed, integrated_data

def save_processed_data(df, filename):
    output_path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path

def main():
    protac_csv = pd.read_csv(os.path.join(DATA_DIR, "protac.csv"), low_memory=False)
    protacdb_csv = pd.read_csv(os.path.join(DATA_DIR, "protacdb_20220210.csv"))
    
    protacdb_processed, integrated_data = integrate_datasets(protac_csv, protacdb_csv)
    
    save_processed_data(protacdb_processed, "protacdb_processed.csv")
    
    with open(os.path.join(RESULTS_DIR, "data_integration_summary.json"), 'w') as f:
        json.dump(integrated_data, f, indent=2)
    
    print("\nData preprocessing complete!")

if __name__ == "__main__":
    main()
