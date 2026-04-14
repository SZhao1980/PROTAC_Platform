import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "data/raw"
RESULTS_DIR = "results"

def load_protac_data():
    protac_csv = pd.read_csv(os.path.join(DATA_DIR, "protac.csv"))
    protacdb_csv = pd.read_csv(os.path.join(DATA_DIR, "protacdb_20220210.csv"))
    print(f"PROTAC.csv: {protac_csv.shape}")
    print(f"PROTACdb.csv: {protacdb_csv.shape}")
    return protac_csv, protacdb_csv

def analyze_data(protacdb_csv):
    stats = {}
    if 'Active/Inactive' in protacdb_csv.columns:
        stats['activity'] = protacdb_csv['Active/Inactive'].value_counts().to_dict()
    if 'Target' in protacdb_csv.columns:
        stats['targets'] = protacdb_csv['Target'].value_counts().head(10).to_dict()
    if 'E3 Ligase' in protacdb_csv.columns:
        stats['e3_ligases'] = protacdb_csv['E3 Ligase'].value_counts().to_dict()
    if 'MW' in protacdb_csv.columns:
        stats['mw'] = {'mean': float(protacdb_csv['MW'].mean()), 'std': float(protacdb_csv['MW'].std())}
    return stats

protac_csv, protacdb_csv = load_protac_data()
stats = analyze_data(protacdb_csv)
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "protac_analysis.json"), 'w') as f:
    json.dump(stats, f, indent=2)
print("Analysis complete")
