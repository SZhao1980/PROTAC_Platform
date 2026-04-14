"""
PROTAC Data Preprocessing Module
Handles data loading, cleaning, and preparation for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProtacDataProcessor:
    """Process PROTAC data from multiple sources"""
    
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_protac_db(self):
        """Load PROTAC-DB data"""
        logger.info("Loading PROTAC-DB data...")
        csv_file = self.data_dir / 'protac.csv'
        
        if not csv_file.exists():
            logger.warning(f"PROTAC-DB CSV not found: {csv_file}")
            return None
            
        df = pd.read_csv(csv_file, low_memory=False)
        logger.info(f"Loaded {len(df)} records from PROTAC-DB")
        return df
    
    def load_protacpedia(self):
        """Load PROTACpedia data"""
        logger.info("Loading PROTACpedia data...")
        csv_file = self.data_dir / 'protacdb_20220210.csv'
        
        if not csv_file.exists():
            logger.warning(f"PROTACpedia CSV not found: {csv_file}")
            return None
            
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} records from PROTACpedia")
        return df
    
    def calculate_molecular_properties(self, smiles):
        """Calculate molecular properties from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            props = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRings': Descriptors.RingCount(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            }
            
            # Lipinski's Rule of Five violations
            props['Lipinski_Violations'] = sum([
                props['MW'] > 500,
                props['LogP'] > 5,
                props['HBA'] > 10,
                props['HBD'] > 5
            ])
            
            return props
        except Exception as e:
            logger.debug(f"Error calculating properties for SMILES: {e}")
            return None
    
    def preprocess_protac_db(self, df):
        """Preprocess PROTAC-DB data"""
        logger.info("Preprocessing PROTAC-DB data...")
        
        # Create a copy
        df = df.copy()
        
        # Extract SMILES column
        df['PROTAC_SMILES'] = df['Smiles']
        
        # Remove rows with missing SMILES
        df = df.dropna(subset=['PROTAC_SMILES'])
        logger.info(f"After removing missing SMILES: {len(df)} records")
        
        # Extract activity labels - mark as active if DC50 is available
        df['Activity'] = 1  # Default to active
        
        # Extract DC50 values if available
        if 'DC50 (nM)' in df.columns:
            df['DC50'] = pd.to_numeric(df['DC50 (nM)'], errors='coerce')
        
        # Extract Dmax values if available
        if 'Dmax (%)' in df.columns:
            df['Dmax'] = pd.to_numeric(df['Dmax (%)'], errors='coerce')
        
        # Extract target protein
        df['Target'] = df['Target'].fillna('Unknown')
        
        # Extract E3 ligase
        df['E3_Ligase'] = df['E3 ligase'].fillna('Unknown')
        
        # Calculate molecular properties
        logger.info("Calculating molecular properties...")
        props_list = []
        for idx, smiles in enumerate(df['PROTAC_SMILES'].values):
            if idx % 1000 == 0:
                logger.info(f"Processing molecule {idx}/{len(df)}")
            props = self.calculate_molecular_properties(smiles)
            props_list.append(props)
        
        props_df = pd.DataFrame(props_list)
        
        # Concatenate with explicit index reset
        df_result = pd.concat([
            df[['PROTAC_SMILES', 'Activity', 'Target', 'E3_Ligase']].reset_index(drop=True),
            props_df
        ], axis=1)
        
        # Remove rows with failed property calculation
        df_result = df_result.dropna(subset=['MW'])
        logger.info(f"After property calculation: {len(df_result)} records")
        
        return df_result
    
    def preprocess_protacpedia(self, df):
        """Preprocess PROTACpedia data"""
        logger.info("Preprocessing PROTACpedia data...")
        
        # Create a copy
        df = df.copy()
        
        # Extract SMILES column
        df['PROTAC_SMILES'] = df['PROTAC SMILES']
        
        # Remove rows with missing SMILES
        df = df.dropna(subset=['PROTAC_SMILES'])
        logger.info(f"After removing missing SMILES: {len(df)} records")
        
        # Extract activity labels
        if 'Active/Inactive' in df.columns:
            df['Activity'] = df['Active/Inactive'].map({'Active': 1, 'Inactive': 0})
            df = df.dropna(subset=['Activity'])
        else:
            df['Activity'] = 1
        
        # Extract DC50 values if available
        if 'Dc50' in df.columns:
            df['DC50'] = pd.to_numeric(df['Dc50'], errors='coerce')
        
        # Extract Dmax values if available
        if 'Dmax' in df.columns:
            df['Dmax'] = pd.to_numeric(df['Dmax'], errors='coerce')
        
        # Extract target protein
        df['Target'] = df['Target'].fillna('Unknown')
        
        # Extract E3 ligase
        df['E3_Ligase'] = df['E3 Ligase'].fillna('Unknown')
        
        # Calculate molecular properties
        logger.info("Calculating molecular properties...")
        props_list = []
        for idx, smiles in enumerate(df['PROTAC_SMILES'].values):
            if idx % 100 == 0:
                logger.info(f"Processing molecule {idx}/{len(df)}")
            props = self.calculate_molecular_properties(smiles)
            props_list.append(props)
        
        props_df = pd.DataFrame(props_list)
        
        # Concatenate with explicit index reset
        df_result = pd.concat([
            df[['PROTAC_SMILES', 'Activity', 'Target', 'E3_Ligase']].reset_index(drop=True),
            props_df
        ], axis=1)
        
        # Remove rows with failed property calculation
        df_result = df_result.dropna(subset=['MW'])
        logger.info(f"After property calculation: {len(df_result)} records")
        
        return df_result
    
    def merge_datasets(self, df_protac_db, df_protacpedia):
        """Merge PROTAC-DB and PROTACpedia datasets"""
        logger.info("Merging datasets...")
        
        # Select common columns
        common_cols = ['PROTAC_SMILES', 'Activity', 'Target', 'E3_Ligase', 
                       'MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'TPSA']
        
        df_protac_db_selected = df_protac_db[
            [col for col in common_cols if col in df_protac_db.columns]
        ].copy()
        df_protac_db_selected['Source'] = 'PROTAC-DB'
        
        df_protacpedia_selected = df_protacpedia[
            [col for col in common_cols if col in df_protacpedia.columns]
        ].copy()
        df_protacpedia_selected['Source'] = 'PROTACpedia'
        
        # Merge datasets with explicit index reset
        df_merged = pd.concat([
            df_protac_db_selected.reset_index(drop=True),
            df_protacpedia_selected.reset_index(drop=True)
        ], ignore_index=True)
        
        logger.info(f"Merged dataset contains {len(df_merged)} records")
        
        return df_merged
    
    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test sets"""
        logger.info("Splitting dataset...")
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_df = df[:train_idx]
        val_df = df[train_idx:val_idx]
        test_df = df[val_idx:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, df_train, df_val, df_test, merged_df):
        """Save processed datasets"""
        logger.info("Saving processed data...")
        
        train_file = self.output_dir / 'train_data.csv'
        val_file = self.output_dir / 'val_data.csv'
        test_file = self.output_dir / 'test_data.csv'
        merged_file = self.output_dir / 'merged_data.csv'
        
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)
        merged_df.to_csv(merged_file, index=False)
        
        logger.info(f"Saved train data to {train_file}")
        logger.info(f"Saved val data to {val_file}")
        logger.info(f"Saved test data to {test_file}")
        logger.info(f"Saved merged data to {merged_file}")
        
        return train_file, val_file, test_file, merged_file
    
    def generate_statistics(self, df):
        """Generate dataset statistics"""
        logger.info("Generating statistics...")
        
        stats = {
            'Total_Records': len(df),
            'Active_Compounds': (df['Activity'] == 1).sum() if 'Activity' in df.columns else 0,
            'Inactive_Compounds': (df['Activity'] == 0).sum() if 'Activity' in df.columns else 0,
            'Avg_MW': float(df['MW'].mean()) if 'MW' in df.columns else None,
            'Avg_LogP': float(df['LogP'].mean()) if 'LogP' in df.columns else None,
            'Avg_TPSA': float(df['TPSA'].mean()) if 'TPSA' in df.columns else None,
            'Unique_Targets': df['Target'].nunique() if 'Target' in df.columns else 0,
            'Unique_E3_Ligases': df['E3_Ligase'].nunique() if 'E3_Ligase' in df.columns else 0,
        }
        
        return stats
    
    def run(self):
        """Run complete preprocessing pipeline"""
        logger.info("Starting PROTAC data preprocessing...")
        
        # Load data
        df_protac_db = self.load_protac_db()
        df_protacpedia = self.load_protacpedia()
        
        if df_protac_db is None or df_protacpedia is None:
            logger.error("Failed to load data")
            return False
        
        # Preprocess
        df_protac_db = self.preprocess_protac_db(df_protac_db)
        df_protacpedia = self.preprocess_protacpedia(df_protacpedia)
        
        if df_protac_db is None or df_protacpedia is None:
            logger.error("Failed to preprocess data")
            return False
        
        # Merge
        df_merged = self.merge_datasets(df_protac_db, df_protacpedia)
        
        # Split
        df_train, df_val, df_test = self.split_dataset(df_merged)
        
        # Save
        self.save_processed_data(df_train, df_val, df_test, df_merged)
        
        # Statistics
        stats = self.generate_statistics(df_merged)
        logger.info(f"Dataset Statistics: {stats}")
        
        # Save statistics
        stats_file = self.output_dir / 'dataset_statistics.txt'
        with open(stats_file, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Preprocessing completed successfully!")
        return True


if __name__ == '__main__':
    processor = ProtacDataProcessor()
    processor.run()
