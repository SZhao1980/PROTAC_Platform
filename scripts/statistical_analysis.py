#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import json
from scipy import stats

DATA_DIR = "data/raw"
RESULTS_DIR = "results"

def load_processed_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "protacdb_processed.csv"))
    return df

def analyze_activity_distribution(df):
    print("\n=== Activity Distribution Analysis ===")
    active = df[df['Activity_Label'] == 1]
    inactive = df[df['Activity_Label'] == 0]
    
    stats_dict = {
        'total': int(len(df)),
        'active': int(len(active)),
        'inactive': int(len(inactive)),
        'active_ratio': float(len(active) / len(df)),
        'inactive_ratio': float(len(inactive) / len(df)),
    }
    
    print(f"Total: {stats_dict['total']}, Active: {stats_dict['active']}, Inactive: {stats_dict['inactive']}")
    return stats_dict, active, inactive

def analyze_target_distribution(df):
    print("\n=== Target Distribution Analysis ===")
    target_dist = df['Target'].value_counts()
    top_targets = target_dist.head(10)
    
    print(f"Total unique targets: {len(target_dist)}")
    for target, count in top_targets.items():
        print(f"  {target}: {count}")
    
    return {
        'total_targets': int(len(target_dist)),
        'top_targets': {str(k): int(v) for k, v in top_targets.to_dict().items()},
        'mean_compounds_per_target': float(target_dist.mean()),
        'std_compounds_per_target': float(target_dist.std()),
    }

def analyze_e3_ligase_distribution(df):
    print("\n=== E3 Ligase Distribution Analysis ===")
    e3_dist = df['E3 Ligase'].value_counts()
    
    for e3, count in e3_dist.items():
        print(f"  {e3}: {count} ({count/len(df):.2%})")
    
    return {
        'e3_distribution': {str(k): int(v) for k, v in e3_dist.to_dict().items()},
        'total_e3_types': int(len(e3_dist)),
    }

def analyze_linker_types(df):
    print("\n=== Linker Type Analysis ===")
    if 'Linker Type' in df.columns:
        linker_dist = df['Linker Type'].value_counts()
        for linker, count in linker_dist.items():
            print(f"  {linker}: {count} ({count/len(df):.2%})")
        return {
            'linker_distribution': {str(k): int(v) for k, v in linker_dist.to_dict().items()},
            'total_linker_types': int(len(linker_dist)),
        }
    return {}

def analyze_molecular_properties(df):
    print("\n=== Molecular Properties Analysis ===")
    properties = {}
    
    if 'MW_numeric' in df.columns:
        mw = df['MW_numeric'].dropna()
        properties['MW'] = {
            'mean': float(mw.mean()),
            'std': float(mw.std()),
            'min': float(mw.min()),
            'max': float(mw.max()),
            'median': float(mw.median()),
        }
        print(f"MW: {properties['MW']['mean']:.2f} ± {properties['MW']['std']:.2f}")
    
    if 'TPSA' in df.columns:
        tpsa = df['TPSA'].dropna()
        properties['TPSA'] = {
            'mean': float(tpsa.mean()),
            'std': float(tpsa.std()),
            'min': float(tpsa.min()),
            'max': float(tpsa.max()),
        }
        print(f"TPSA: {properties['TPSA']['mean']:.2f} ± {properties['TPSA']['std']:.2f}")
    
    if 'Hbond acceptors' in df.columns:
        hba = df['Hbond acceptors'].dropna()
        properties['HBA'] = {
            'mean': float(hba.mean()),
            'std': float(hba.std()),
        }
    
    if 'Hbond donors' in df.columns:
        hbd = df['Hbond donors'].dropna()
        properties['HBD'] = {
            'mean': float(hbd.mean()),
            'std': float(hbd.std()),
        }
    
    return properties

def analyze_activity_vs_properties(df):
    print("\n=== Activity vs Molecular Properties ===")
    active = df[df['Activity_Label'] == 1]
    inactive = df[df['Activity_Label'] == 0]
    comparisons = {}
    
    if 'MW_numeric' in df.columns:
        active_mw = active['MW_numeric'].dropna()
        inactive_mw = inactive['MW_numeric'].dropna()
        if len(active_mw) > 0 and len(inactive_mw) > 0:
            t_stat, p_value = stats.ttest_ind(active_mw, inactive_mw)
            comparisons['MW'] = {
                'active_mean': float(active_mw.mean()),
                'inactive_mean': float(inactive_mw.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
            }
            print(f"MW: p={p_value:.4f}")
    
    if 'TPSA' in df.columns:
        active_tpsa = active['TPSA'].dropna()
        inactive_tpsa = inactive['TPSA'].dropna()
        if len(active_tpsa) > 0 and len(inactive_tpsa) > 0:
            t_stat, p_value = stats.ttest_ind(active_tpsa, inactive_tpsa)
            comparisons['TPSA'] = {
                'active_mean': float(active_tpsa.mean()),
                'inactive_mean': float(inactive_tpsa.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
            }
            print(f"TPSA: p={p_value:.4f}")
    
    return comparisons

def save_comprehensive_analysis(all_results):
    output_file = os.path.join(RESULTS_DIR, "comprehensive_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAnalysis saved to {output_file}")

def main():
    print("Starting comprehensive statistical analysis...")
    df = load_processed_data()
    
    all_results = {}
    activity_stats, active_df, inactive_df = analyze_activity_distribution(df)
    all_results['activity_distribution'] = activity_stats
    
    target_stats = analyze_target_distribution(df)
    all_results['target_distribution'] = target_stats
    
    e3_stats = analyze_e3_ligase_distribution(df)
    all_results['e3_ligase_distribution'] = e3_stats
    
    linker_stats = analyze_linker_types(df)
    all_results['linker_analysis'] = linker_stats
    
    mol_props = analyze_molecular_properties(df)
    all_results['molecular_properties'] = mol_props
    
    comparisons = analyze_activity_vs_properties(df)
    all_results['activity_property_comparison'] = comparisons
    
    save_comprehensive_analysis(all_results)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
