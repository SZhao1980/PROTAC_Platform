"""
Simplified PROTAC Activity Prediction Model
Using molecular descriptors and machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MolecularDescriptorCalculator:
    """Calculate molecular descriptors from SMILES"""
    
    @staticmethod
    def calculate_descriptors(smiles):
        """Calculate comprehensive molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRings': Descriptors.RingCount(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'NumAliphaticCycles': Descriptors.NumAliphaticCycles(mol),
                'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'FractionCsp3': Descriptors.FractionCsp3(mol),
                'Ipc': Descriptors.Ipc(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'PercentRotatableBonds': Descriptors.PercentRotatableBonds(mol),
                'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
                'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
            }
            
            return descriptors
        except Exception as e:
            logger.debug(f"Error calculating descriptors: {e}")
            return None


class ProtacActivityPredictor:
    """PROTAC Activity Prediction Model"""
    
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _create_model(self):
        """Create the prediction model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, smiles_list):
        """Prepare features from SMILES"""
        logger.info(f"Calculating descriptors for {len(smiles_list)} molecules...")
        
        descriptor_calc = MolecularDescriptorCalculator()
        descriptors_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            if idx % 1000 == 0:
                logger.info(f"Processing molecule {idx}/{len(smiles_list)}")
            
            descriptors = descriptor_calc.calculate_descriptors(smiles)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                valid_indices.append(idx)
        
        if not descriptors_list:
            raise ValueError("No valid molecules found")
        
        df_descriptors = pd.DataFrame(descriptors_list)
        self.feature_names = df_descriptors.columns.tolist()
        
        logger.info(f"Successfully calculated descriptors for {len(descriptors_list)} molecules")
        
        return df_descriptors.values, np.array(valid_indices)
    
    def train(self, train_csv):
        """Train the model"""
        logger.info("Loading training data...")
        train_df = pd.read_csv(train_csv)
        
        # Prepare features
        X, valid_idx = self.prepare_features(train_df['PROTAC_SMILES'].values)
        y = train_df['Activity'].values[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train model
        logger.info("Training model...")
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        train_metrics = {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0)
        }
        
        logger.info(f"Training metrics: {train_metrics}")
        
        return train_metrics
    
    def evaluate(self, test_csv):
        """Evaluate the model"""
        logger.info("Loading test data...")
        test_df = pd.read_csv(test_csv)
        
        # Prepare features
        X, valid_idx = self.prepare_features(test_df['PROTAC_SMILES'].values)
        y = test_df['Activity'].values[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        test_metrics = {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0)
        }
        
        logger.info(f"Test metrics: {test_metrics}")
        
        return test_metrics, y_pred_proba, y
    
    def save(self, model_path):
        """Save the model"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path):
        """Load the model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        logger.info(f"Model loaded from {model_path}")


def train_and_evaluate_models(train_csv, val_csv, test_csv, output_dir='models'):
    """Train and evaluate multiple models"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_type in ['random_forest', 'gradient_boosting']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type} model...")
        logger.info(f"{'='*60}\n")
        
        predictor = ProtacActivityPredictor(model_type=model_type)
        
        # Train
        train_metrics = predictor.train(train_csv)
        
        # Evaluate on validation set
        val_metrics, val_pred_proba, val_y = predictor.evaluate(val_csv)
        
        # Evaluate on test set
        test_metrics, test_pred_proba, test_y = predictor.evaluate(test_csv)
        
        # Save model
        model_path = output_path / f'{model_type}_model.pkl'
        predictor.save(model_path)
        
        results[model_type] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_predictions': val_pred_proba,
            'test_predictions': test_pred_proba
        }
    
    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_AUC': [results[m]['train_metrics']['auc'] for m in results.keys()],
        'Val_AUC': [results[m]['val_metrics']['auc'] for m in results.keys()],
        'Test_AUC': [results[m]['test_metrics']['auc'] for m in results.keys()],
        'Test_Accuracy': [results[m]['test_metrics']['accuracy'] for m in results.keys()],
        'Test_F1': [results[m]['test_metrics']['f1'] for m in results.keys()],
    })
    
    results_df.to_csv(output_path / 'model_comparison.csv', index=False)
    logger.info(f"\nModel comparison:\n{results_df}")
    
    return results


if __name__ == '__main__':
    train_csv = 'data/processed/train_data.csv'
    val_csv = 'data/processed/val_data.csv'
    test_csv = 'data/processed/test_data.csv'
    
    if Path(train_csv).exists() and Path(val_csv).exists() and Path(test_csv).exists():
        results = train_and_evaluate_models(train_csv, val_csv, test_csv)
        logger.info("\nModel training and evaluation completed!")
    else:
        logger.error("Data files not found. Please run data preprocessing first.")
